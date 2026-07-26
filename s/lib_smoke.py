import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
      import tensorflow as tf
      print("tf", tf.__version__)
      print("tf devices", tf.config.list_physical_devices())
except Exception as e:
      print("no tf? ", str(e))

try:
      import keras
      print("keras", keras.__version__)
except Exception as e:
      print("no keras? ", str(e))

try:
      import keras_cv
      print("keras_cv", keras_cv.__version__)
except Exception as e:
      print("no keras_cv? ", str(e))

try:
      import jax, jaxlib
      import jax.numpy as jnp
      print("jax", jax.__version__, "jaxlib", jaxlib.__version__)
      print("jax devices", jax.devices())
      print(">test_mat_mul")
      @jax.jit
      def test_mat_mul(a, b):
            return a*b
      x = jnp.array([1,2,3])
      y = jnp.array([4,5,6])
      if test_mat_mul(x, y).shape != (3,):
            raise Exception("jax mat mul failed")
except Exception as e:
      print("no jax? ", str(e))

print("LGTM")
