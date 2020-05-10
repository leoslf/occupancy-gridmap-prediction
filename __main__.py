from gridmap.unet import *
from gridmap.residual_fully_conv_vae import *
from gridmap.gan import *
import pickle

import argparse

models = [
    "ResidualFullyConvVAE",
    # "GAN",
    # "UNet",
]

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Train and/or run models.")
parser.add_argument("--save-history", dest="save_history", default=False, action="store_true",
                   help="save history")
parser.add_argument("--evaluate", dest="evaluate", default=False, action="store_true",
                   help="evaluate the model")
argv = parser.parse_args()

def get_model(name, *argv, **kwargs):
    return globals()[name](*argv, **kwargs)

def handle_history(model, history, test_metric, test_metric_value):
    if not os.path.exists("outputs"):
        os.mkdir("outputs")

    for metric, val_metric in map(tee_val, metrics):
        fig, ax = plt.subplots(1)

        ax.plot(history.history[metric], label = "Train")
        ax.plot(history.history[val_metric], label = "Validation")
        ax.set_title("Model: %s - %s" % (model.name, metric))
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)

        fig.legend(loc = "upper right")
        fig.savefig("outputs/%(model_name)s_%(metric)s_%(test_metric)s_%(value).4f.png" % dict(model_name = model.name, metric = metric, test_metric = test_metric, value = test_metric_value))
        
        plt.close(fig)

if __name__ == "__main__":
    # Muting PIL
    logging.getLogger("PIL").setLevel(level=logging.ERROR)
    # Muting Tensorflow warning
    logging.getLogger("tensorflow").setLevel(level=logging.ERROR)
    logging.getLogger("matplotlib").setLevel(level=logging.ERROR)

    logging.basicConfig(level=logging.DEBUG)

    # losses = {}

    tee_val = lambda metric: (metric, "val_" + metric)


    for model in map(get_model, models):
        metrics = ["loss"] + model.metrics

        # Training
        history = model.fit_df()

        if argv.evaluate:
            test_metrics = model.evaluate_df()

            test_loss = test_metrics[0]

            logger.info("model \"%s\": testing loss: %f", model.name, test_loss)

            # losses[model.name] = dict(zip(metrics, test_metrics))

            if argv.save_history:
                handle_history(model, history, "loss", test_loss)

    # with open("losses.pickle", "wb") as f:
    #     pickle.dump(losses, f)
    
