"""把六组参数的候选更新与执行格式转换放进统一 CHIP 入口。"""

from .training_pipeline import bind
from .training_programs import make_adam
from .training_optimizer_programs import make_publish


def make_pipeline(shapes, *, update):
    parameters, args, body, functions = [], [], [], {}
    outputs = []
    for i, (name, shape) in enumerate(shapes.items()):
        names = [f"p{i}_{label}" for label in
                 (("master", "gradient", "first", "second", "next", "next_first", "next_second", "execution")
                  if update else ("master", "execution"))]
        for j, value in enumerate(names):
            is_output = j >= 4 if update else j == 1
            parameters.append(f"{value}: pl.Out[pl.Tensor]" if is_output else f"{value}: pl.Tensor")
        args += names
        functions[f"publish_{i}"] = make_publish(shape, gamma="gamma" in name, inline=True)
        if update:
            functions[f"adam_{i}"] = make_adam(shape, inline=True)
            body.append(f"    {', '.join(names[4:7])} = adam_{i}({', '.join(names[:7])}, rate, correction1, correction2, clip)")
            master = names[4]
            outputs += names[4:7]
        else:
            master = names[0]
        body.append(f"    {names[-1]} = publish_{i}({master}, {names[-1]})")
        outputs.append(names[-1])
    if update:
        for name in ("rate", "correction1", "correction2", "clip"):
            parameters.append(f"{name}: pl.Scalar[pl.FP32]")
            args.append(name)
    prefix = "adam_all" if update else "render_all"
    text = "\n".join([f"def {prefix}_chip({', '.join(parameters)}):", *body,
        "    return "+", ".join(outputs), f"def {prefix}_host({', '.join(parameters)}):",
        f"    {prefix}_chip({', '.join(args)}, device=0)"])+"\n"
    return bind(text, functions, prefix)
