import argparse
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import utils.metrics
import utils.losses


class SupervisedForecastTask(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        regressor="linear",
        loss="mse",
        args=None,
        pre_len: int = 3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1.5e-3,
        feat_max_val: float = 1.0,
        **kwargs
    ):
        super(SupervisedForecastTask, self).__init__()
        self.save_hyperparameters()
        self.model = model
        self.regressor = (
            nn.Linear(
                self.model.hyperparameters.get("hidden_dim")
                or self.model.hyperparameters.get("output_dim"),
                self.hparams.pre_len,
            )
            if regressor == "linear"
            else regressor
        )
        self.args=args
        self._loss = loss
        self.feat_max_val = feat_max_val

    def forward(self, x):
        # (batch_size, seq_len, num_nodes)
        if len(x.shape)==4:
            batch_size, _, num_nodes,_ = x.size()
        else:
            batch_size, _, num_nodes = x.size()
        # (batch_size, num_nodes, hidden_dim)
        hidden = self.model(x)
        hidden_mean,hidden_var=torch.chunk(hidden,chunks=2,dim=-1)
        #print("mean",hidden_mean.mean())
        #print("var",hidden_var.mean())
        rand=torch.rand_like(hidden_var)
        hidden=hidden_mean+(torch.sqrt(hidden_var+1e-10)*rand)
        #print("hidden",hidden.mean())
        # (batch_size * num_nodes, hidden_dim)
        hidden = hidden.reshape((-1, hidden.size(2)))
        # (batch_size * num_nodes, pre_len)
        if self.regressor is not None:
            predictions = self.regressor(hidden)
        else:
            predictions = hidden
        predictions = predictions.reshape((batch_size, num_nodes, -1))
        return predictions
    def forward_val(self, x):
        # (batch_size, seq_len, num_nodes)
        if len(x.shape)==4:
            batch_size, _, num_nodes,_ = x.size()
        else:
            batch_size, _, num_nodes = x.size()
        # (batch_size, num_nodes, hidden_dim)
        hidden = self.model(x)
        hidden_mean,hidden_var=torch.chunk(hidden,chunks=2,dim=-1)
        #rand=torch.rand_like(hidden_var)
        hidden=hidden_mean
        # (batch_size * num_nodes, hidden_dim)
        hidden = hidden.reshape((-1, hidden.size(2)))
        # (batch_size * num_nodes, pre_len)
        if self.regressor is not None:
            predictions = self.regressor(hidden)
        else:
            predictions = hidden
        predictions = predictions.reshape((batch_size, num_nodes, -1))
        return predictions
    def val_step(self, batch, batch_idx):
        # (batch_size, seq_len/pre_len, num_nodes)
        x, y = batch
        #print(y.shape)
        num_nodes = x.size(2)
        predictions = self.forward_val(x)
        predictions = predictions.transpose(1, 2).reshape((-1, num_nodes))
        y = y.reshape((-1, y.size(2)))
        return predictions, y
    def noise_data(self,x):
        mask=torch.rand(x.shape).to(x.device)
        mask=mask<self.args.noise_ratio
        max_value=torch.max(x)-torch.min(x)

        noise1=torch.distributions.Uniform(low=-1, high=1).sample(x[mask].shape).to(x.device)
        x[mask]=x[mask]+noise1*max_value*self.args.noise_sever*0.1
        return x

    def shared_step(self, batch, batch_idx):
        # (batch_size, seq_len/pre_len, num_nodes)

        x, y = batch
        if self.args.noise==1:
            x=self.noise_data(x)
        num_nodes = x.size(2)
        predictions = self(x)
        predictions = predictions.transpose(1, 2).reshape((-1, num_nodes))
        y = y.reshape((-1, y.size(2)))
        return predictions, y

    def loss(self, inputs, targets):
        if self._loss == "mse":
            mean=self.model.tgcn_cell.mean
            var=self.model.tgcn_cell.var
            kl_loss=torch.mean(torch.square(mean)+var-torch.log(1e-10+var)-1,-1)*0.5
            kl_loss=torch.sum(kl_loss)
            print("kl loss",kl_loss.mean())
            return F.mse_loss(inputs, targets)+self.hparams.kl_gamma*kl_loss
        if self._loss == "mse_with_regularizer":
            mean=self.model.tgcn_cell.mean
            var=self.model.tgcn_cell.var
            kl_loss=torch.mean(torch.square(mean)+var-torch.log(1e-10+var)-1,-1)*0.5
            kl_loss=torch.sum(kl_loss)
            return utils.losses.mse_with_regularizer_loss(inputs, targets, self,self.hparams.weight_decay)+self.hparams.kl_gamma*kl_loss
        raise NameError("Loss not supported:", self._loss)

    def training_step(self, batch, batch_idx):

        predictions, y = self.shared_step(batch,batch_idx)
        loss = self.loss(predictions, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        predictions, y = self.val_step(batch, batch_idx)
        predictions = predictions * self.feat_max_val
        y = y * self.feat_max_val
        loss = self.loss(predictions, y)
        rmse = torch.sqrt(torchmetrics.functional.mean_squared_error(predictions, y))
        mae = torchmetrics.functional.mean_absolute_error(predictions, y)
        accuracy = utils.metrics.accuracy(predictions, y)
        r2 = utils.metrics.r2(predictions, y)
        explained_variance = utils.metrics.explained_variance(predictions, y)
        metrics = {
            "val_loss": loss,
            "RMSE": rmse,
            "MAE": mae,
            "accuracy": accuracy,
            "R2": r2,
            "ExplainedVar": explained_variance,
        }
        self.log_dict(metrics)
        return predictions.reshape(batch[1].size()), y.reshape(batch[1].size())

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
        )

    @staticmethod
    def add_task_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", "--lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", "--wd", type=float, default=5e-4)
        parser.add_argument("--loss", type=str, default="mse")
        parser.add_argument("--lamda", type=float, default=1.0)
        parser.add_argument("--noise", type=int, default=1,choices=[0,1,2])
        parser.add_argument("--noise_ratio", type=float, default=0.2)
        parser.add_argument("--noise_ratio_node",type=float,default=0.2)
        parser.add_argument("--noise_test", action='store_true')
        parser.add_argument("--noise_ratio_test", type=float, default=0.2)
        parser.add_argument("--noise_ratio_node_test",type=float,default=0.2)
        parser.add_argument("--noise_sever", type=float, default=1.0)
        parser.add_argument("--noise_type", type=str, default='gaussian',choices=['gaussian','gaussian_neighbor','missing_neighbor','missing'])
        parser.add_argument("--kl_gamma", type=float, default=5e-4)
        return parser
