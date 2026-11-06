# AGENTS 使用说明
## 本项目修改约束

- 若用户没有明确要求在主程序中新增或删除类/函数（例如仅要求修改某个方法），不得擅自新增或删除类/函数。
- 类和函数注释格式参照 `class Population`：简介、参数、返回、示例（有必要时）。
- 约束：包含 DeepMD 计算器的 `Atoms` 对象不可深拷贝
- 测试约束：默认使用轻量测试（小规模、少代数），避免运行高计算量基准或长时间任务；仅在用户明确要求时执行重负载测试。
- 测试脚本统一放在 `./test/` 下，并在该目录内运行。测试时将标准输出重定向到 `./test/stdout`
- 此后的注释和文档使用英文书写
- 避免访问和修改工作目录之外的文件, 除非为了启动 conda 环境
- 每次修改后给一个简短 Git 提交说明供用户参考(标点用 ASCII字符), 禁止提交
- 每次修改版本号最后一位+1, 如0.0.1->0.0.2, 不要改前2位

## Typing Rules

- Use Python 3.13 style type hints for all new code.
- Add explicit type annotations for variables whenever reasonable, especially:
  - function local variables with non-trivial inferred types
  - return values of helper computations
  - dictionaries, lists, tuples, and numpy arrays
- Do not add a type annotation when the type is obvious from the right-hand side,
  for example `element = str(...)`, `path = Path(...)`, `count = len(...)`.
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
- If a function returns `None`, do not add a `Returns` section.
- For extremely simple functions, a single-line summary docstring is enough;
  do not expand it into structured sections.
- Class docstrings should include key Attributes.
- Use ASCII punctuation in comments and docstrings in the main program.
  Do not use full-width or other Unicode punctuation marks.

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
- Use `pathlib` for all path handling.

## Linting Rules

- Use ruff to lint modified code; see the `ruff` skill for usage details.
