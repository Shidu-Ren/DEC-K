from .videomme import eval_videomme


def _missing_eval(name, error):
    def _raise(*args, **kwargs):
        raise ImportError(f"{name} evaluator is unavailable: {error}") from error

    return _raise


try:
    from .videommmu import eval_videommmu
except ImportError as exc:
    eval_videommmu = _missing_eval("videommmu", exc)

try:
    from .longvideobench import eval_longvideobench, generate_submission_longvideobench
except ImportError as exc:
    eval_longvideobench = _missing_eval("longvideobench", exc)
    generate_submission_longvideobench = _missing_eval("longvideobench submission", exc)

try:
    from .cinepile import eval_cinepile
except ImportError as exc:
    eval_cinepile = _missing_eval("cinepile", exc)

try:
    from .mlvu import eval_mlvu
except ImportError as exc:
    eval_mlvu = _missing_eval("mlvu", exc)

try:
    from .mmvu import eval_mmvu
except ImportError as exc:
    eval_mmvu = _missing_eval("mmvu", exc)

try:
    from .mmworld import eval_mmworld
except ImportError as exc:
    eval_mmworld = _missing_eval("mmworld", exc)

try:
    from .hourvideo import generate_submission_hourvideo
except ImportError as exc:
    generate_submission_hourvideo = _missing_eval("hourvideo submission", exc)

try:
    from .egolife import eval_egolife
except ImportError as exc:
    eval_egolife = _missing_eval("egolife", exc)

try:
    from .cgbench import eval_cgbench, eval_cgbench_miou
except ImportError as exc:
    eval_cgbench = _missing_eval("cgbench", exc)
    eval_cgbench_miou = _missing_eval("cgbench-miou", exc)

try:
    from .videommlu import eval_videommlu
except ImportError as exc:
    eval_videommlu = _missing_eval("videommlu", exc)

try:
    from .minerva import eval_minerva
except ImportError as exc:
    eval_minerva = _missing_eval("minerva", exc)

try:
    from .m3bench import eval_m3bench
except ImportError as exc:
    eval_m3bench = _missing_eval("m3bench", exc)
