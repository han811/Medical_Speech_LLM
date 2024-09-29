from absl import app, flags

from core.trainer import Trainer

FLAGS = flags.FLAGS

flags.DEFINE_string("log_path", None, "Log file path")
flags.DEFINE_integer("gpus", 1, "# of gpus")


def main(argv):
    trainer = Trainer(log_path=FLAGS.log_path, gpus=FLAGS.gpus)
    trainer.train()


if __name__ == "__main__":
    app.run(main)
