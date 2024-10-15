#!/usr python

from gcnmonitor import handler
import gcn
# import gcn_custom as gcn

gcn.listen(handler=handler)
