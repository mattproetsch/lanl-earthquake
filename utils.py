from time import strftime


def clean_varname(varname):
    varname = re.sub(r'[^A-Za-z0-9]', '_', varname)
    varname = re.sub(r'^_', '', varname)
    varname = re.sub(r'_$', '', varname)
    varname = re.sub(r'__+', '_', varname)
    return varname


class Loggable(object):

    def _log(self, layer_name, var_name, tens):
        if not self._DEBUG:
            return

        description = '(' + str(tens.dtype.name) + ' '
        sizes = tens.get_shape()
        for i, size in enumerate(sizes):
            description += str(size)
            if i < len(sizes) - 1:
                description += 'x'
        description += ')'
        msg = '[%s] [%s]: %s %s' % (strftime('%Y-%m-%d %H:%M:%S'), layer_name, var_name, description)
        print(msg)
