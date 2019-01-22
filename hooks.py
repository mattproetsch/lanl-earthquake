import tensorflow as tf

class ModelStepTrackerHook(tf.train.SessionRunHook):
    """
    Adds a global step counter to the graph. Same as global_step, except for 2 things:
    1) Available during training, evaluation, and test.
    2) Persistent only during the entire call to tf.estimator.train, .eval, and .predict
    Increments the step tracker by one for each call to `sess.run`.
    timeline here is helpful: https://medium.com/@tijmenlv/an-advanced-example-of-tensorflow-estimators-part-3-3-8c2efe8ff6fa
    """
    def __init__(self):
        self._ops_added = False
        self._steps_tensor = None
        self._steps_placeholder = None
        self._assign_op = None
        self._step_needs_update = False
        self._steps = 0
        
    def _update_step(self, sess):
        """Run the assign op, setting the step_counter/step_counter tensor to self._steps"""
        assert self._ops_added
        sess.run([self._assign_op], feed_dict = {self._steps_placeholder: self._steps})
        self._step_needs_update = False
    
    def begin(self):
        self._steps = 0
        if not self._ops_added:
            with tf.variable_scope('step_counter', reuse=tf.AUTO_REUSE):
                self._steps_tensor = tf.get_variable('step_counter',
                                                     shape=[],
                                                     dtype=tf.int64,
                                                     trainable=False)
                self._steps_placeholder = tf.placeholder(dtype=tf.int64,
                                                         shape=[],
                                                         name='step_counter_placeholder')
                self._assign_op = tf.assign(self._steps_tensor, self._steps_placeholder)
            self._ops_added = True
        self._step_needs_update = True
    
    def before_run(self, run_context):
        if self._step_needs_update:
            self._update_step(run_context.session)
    
    def after_run(self, run_context, run_values):
        self._steps += 1
        self._step_needs_update = True
