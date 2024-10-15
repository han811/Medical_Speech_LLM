import gin
from absl import app, flags

from core.task import Task
from core.trainer import TrainerWrapper

FLAGS = flags.FLAGS

flags.DEFINE_string("log_path", "./logs", "Log file path")
flags.DEFINE_string("exp_name", "voice_llm", "Experiment name for logger")
flags.DEFINE_integer("gpus", 1, "# of gpus")

flags.DEFINE_string("task_gin", "./gin/task/base.gin", "path to task gin file")
flags.DEFINE_string("model_gin", "./gin/model/base.gin", "path to model gin file")


def main(argv):
    gin.parse_config_file(FLAGS.task_gin)
    gin.parse_config_file(FLAGS.model_gin)

    task = Task()
    trainer = TrainerWrapper(
        task=task,
        log_path=FLAGS.log_path,
        exp_name=FLAGS.exp_name,
        gpus=FLAGS.gpus,
    )
    trainer.train()


if __name__ == "__main__":
    app.run(main)
