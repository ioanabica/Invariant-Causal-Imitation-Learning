import os
import sys

import tensorflow as tf

sys.path.append(os.path.join(os.getcwd(), "contrib", "baselines_zoo"))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from contrib.baselines_zoo import (  # pylint: disable=unused-import # noqa: F401, E402
    enjoy as oastable,
)


class ArgParse(dict):
    def __init__(self, *args, **kwargs):
        super(ArgParse, self).__init__(*args, **kwargs)
        self.__dict__ = self
