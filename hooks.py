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
        

class SlimVarAnalyzer(tf.train.SessionRunHook):
    """
    Uses two functions from tf.contrib.slim@83d33cc on Dec 5, 2016
    tf.contrib.slim.model_analyzer.analyze_vars, and
    tf.contrib.slim.model_analyzer.tensor_description
    """
    
    def __init__(self):
        pass
    
    def _tensor_description(self, var):
        """
        Returns a compact and informative string about a tensor.
        Args:
            var: A tensor variable.
        Returns:
            a string with type and size, e.g.: (float32 1x8x8x1024).
        """
        description = '(' + str(var.dtype.name) + ' '
        sizes = var.get_shape()
        try:
            for i, size in enumerate(sizes):
                description += str(size)
                if i < len(sizes) - 1:
                    description += 'x'
        except:
            description += 'unknown size'
        description += ')'
        
        return description
        
        
    def _analyze_vars(self, variables):
        """
        Prints the names and shapes of the variables.
        Args:
            variables: list of variables, for example tf.global_variables().
            print_info: Optional, if true print variables and their shape.
        Returns:
            A printable string with model parameter information
        """
        
        s = ''
        s += '----------\n'
        s += 'Variables: name (type shape) [size]\n'
        s += '----------\n'
        
        total_size = 0
        total_bytes = 0
        for var in variables:
            # if var.num_elements() is None or [] assume size 0.
            var_size = var.get_shape().num_elements() or 0
            var_bytes = var_size * var.dtype.size
            total_size += var_size
            total_bytes += var_bytes
            s += '%s %s [%d, bytes: %d]\n' % (var.name, self._tensor_description(var),
                                                var_size, var_bytes)
        s += 'Total size of variables: %d\n' % total_size
        s += 'Total bytes of variables: %d\n' % total_bytes
        return s
                
    def _model_summary(self):
        model_vars = tf.trainable_variables()
        return self._analyze_vars(model_vars)
    
    #def before_run(self, run_context):
        #print(self._model_summary())
    
    def begin(self):
        print(self._model_summary())