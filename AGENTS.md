# AGENTS 使用说明
## 代码修改约束（SaNSDE_new.py）

- 若用户没有明确要求在主程序 `SaNSDE_new.py` 中新增或删除类/函数（例如仅要求修改某个方法），不得擅自新增或删除类/函数。
- 自定义变量名优先使用驼峰命名法。
- 类和函数注释格式参照 `class Population`：简介、参数、返回、示例（有必要时）。
- 主程序中的注释和 docstring 标点统一使用 ASCII 英文标点符号。
- 类型注解使用 Python 新版本规范（例如 `list[float | None]`，避免 `List[Optional[float]]` 旧写法）。
- 代码实现与类型写法以 Python 3.13 规范为准。
- 约束：包含 DeepMD 计算器的 `Atoms` 对象不可深拷贝；`localOptimize` 优化结束后必须清除 `atoms.calc`，并避免将带计算器对象的团簇进入后续种群深拷贝流程。
- 测试约束：默认使用轻量测试（小规模、少代数），避免运行高计算量基准或长时间任务；仅在用户明确要求时执行重负载测试。
- 进行程序测试时，先备份主程序到`./test/`, 然后必须在 `./test/` 下新建独立 Python 脚本并 `import SaNSDE_new`，不得将测试脚本写入主程序。
- 测试时将标准输出重定向到 `./stdout/`，文件命名格式：`YYYYMMddhhmmss-任务名.txt`。

# 新代码规范
## Typing Rules

- Use Python 3.13 style type hints for all new code.
- Add explicit type annotations for variables whenever reasonable, especially:
  - function local variables with non-trivial inferred types
  - return values of helper computations
  - dictionaries, lists, tuples, and numpy arrays
- Prefer built-in generic syntax:
  - `list[int]`, `dict[str, float]`, `tuple[int, int]`
- Prefer union operator syntax:
  - `str | None` instead of `Optional[str]`
- Keep aliases explicit when needed:
  - `type CitationMap = dict[np.int16, npt.NDArray[np.int16]]`

## Documentation Rules

- All public classes and functions must have docstrings.
- Public methods should also have docstrings.
- Special methods that are not typically called directly (for example `__len__`, `__iter__`, `__getitem__`) do not need per-method docstrings.
- Put usage of such special methods in the class docstring `Examples` section.
- Follow PEP 257 + common structured docstring convention (Google-style sections).
- Each public API docstring should include:
  - brief summary
  - Args
  - Returns
  - Raises (if applicable)
  - Examples (when helpful)
- Class docstrings should include key Attributes.

## Naming Rules

- Use unified, concise, and clear naming.
- Use `snake_case` for variables and functions.
- Use `PascalCase` for classes.
- Avoid ambiguous abbreviations unless they are domain-standard (for example `if`).
- Keep naming consistent across parsing, model, and calculation logic.

## Coding style

- Use EAFP style, for example:
  ```python
  # EAFP style
  try:
      x = my_dict['key']
  except KeyError:
      # handle missing key

  # LBYL style
  if 'key' in my_dict:
      x = my_dict['key']
  else:
      # handling missing key
  ```