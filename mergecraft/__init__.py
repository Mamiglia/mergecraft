# from .computation.hessian import hess
# from .computation.hf_extension import HessianCallback, add_callback
from .arithmetics.weights_wrapper import StateDict, dict_map
from .merging.base import model_merge, layer_merge
from .merging.soup import soup
from .merging.stock import stock
from .merging.ties import ties
from .merging.dare import dare
from .merging.task import task
from .merging.fisher import fisher
from .merging.slerp import slerp

from .evaluation import evaluate_glue_pipeline
