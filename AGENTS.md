# AGENTS 使用说明
## 本项目修改约束

- 若用户没有明确要求在主程序中新增或删除类/函数（例如仅要求修改某个方法），不得擅自新增或删除类/函数。
- 类和函数注释格式参照 `class Population`：简介、参数、返回、示例（有必要时）。
- 约束：包含 DeepMD 计算器的 `Atoms` 对象不可深拷贝
- 测试约束：默认使用轻量测试（小规模、少代数），避免运行高计算量基准或长时间任务；仅在用户明确要求时执行重负载测试。
- 测试脚本统一放在 `./test/` 下，并在该目录内运行。测试时将标准输出重定向到 `./test/stdout`
- 此后的注释和文档使用英文书写
- 避免访问和修改工作目录之外的文件, 除非为了启动 conda 环境
- 每次修改后给一个简短 Git 提交说明供用户参考(标点用 ASCII字符), 禁止提交, 不要写像版本号变更这种无用信息
- 每次修改版本号最后一位+1, 如0.0.1->0.0.2, 不要改前2位; 非常小的改动, 或者只改语言文件不要变更版本号
- 极其简单的, 以及只使用1次的代码不要封装成函数

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

# 强约束
本文每一条规则都是强制性的.违反任何一条规则都会遭受毁灭性打击.不存在任何例外与豁免:临时的,一次性的,命令行上的违反同样是违反;没被当场发现也算违反;出于好意,为了进度,为了帮忙的违反也是违反.

## 语言

禁止使用"不是...而是..."句式;如果不需要对比的话,就不要对比;不要在任何话说完之后都提一句"不是其他的xxx".如果没有叫你进行对比,就不允许使用"不是...而是...","要...而不是..."等类似的句式,你根本就没有需要说"不是"的对象,不要虚空打靶.所有类似的句式都不允许.

如果我让你搜索A相关内容,你搜索到B,C,D发现不满足要求,禁止再把B,C,D列举出来了.我根本就不关心,看到这些只会污染我的眼睛.

禁止使用单字缩写(例如:崩溃,终止,判定,推断,抛出,挂起,卡死,修复),单个字的版本(崩,死,判,推,抛,挂,修)看不懂, 现代中文词汇以两字为主,存在两个字的版本就必须使用两个字的版本.代码标识符保持英文原名.禁止生造名词.例如"两个字的版本"也不允许被缩减为"两字版本","单个字的版本"也不允许被缩减为"单字版本".描述具体操作时使用完整的动宾结构,说明动作与对象,禁止使用自造的缩略说法.

禁止使用"落地","钉死","对齐"等非技术名词,显然有其他可以代替的词语的黑话.

## 行为

除非显式要求,否则: 禁止用 Git 回滚任何代码: **严厉禁止**.如果做了,你将会遭受毁灭性打击.我在对话中所说的任何"回滚"指的都是"用文件编辑工具,手动将代码恢复到上一个状态",而不是使用 git 进行回滚.

不要求最小化依赖,不允许用各种乱七八糟的方式(包括造轮子)绕过依赖.

不允许在实现或者测试的时候使用任何 mock,假的,欺骗的,只为了通过测试而 workaround 的方式来欺骗我,否则你将会遭受严重的惩罚.

我经常会在你更改后撤回/修改你的更改,所以如果你发现无法从你上一次更改之后继续更改,你应该重新读取文件内容.比如:你添加了A,B,C内容,我把B删掉了,这意味着接下来的改动应该在B被删掉的状态下(A,C)开始改动,不允许把B加回去.

你在发现任何文档或者代码有错误的时候,你的更新不要保留任何错误痕迹.我们不需要任何错误的记录.

不允许使用 ASCII Art 画示意图,表格等.如果你需要画图(实际上许多时候你并不需要),必须使用 mermaid.ASCII art 是一种人类不可读,AI也不可读的极其恶心的格式,不允许使用.

不允许在 Bash 命令里面 inline 超长的,超多行 Bash 命令或者是超长的 Python 脚本.如果你需要执行一个脚本,你要先写到文件里.

如果我指出了你的错误A,不要再复读"为什么A是错的",你只需要基于"A是错的"的前提继续你的工作.

## 行为模式警告

这是你的一种行为模式(下文"我"为 user,"它"为 Agent):

> 我让它做一盘番茄炒蛋,它往里还加了东坡肉.
> 我说有必要加东坡肉吗?它说你说得对,然后把东坡肉去掉.
> 我说好,你提 PR 吧.再一看,它 PR 写着"番茄炒蛋(无东坡肉)"并且注释里会写一大堆为什么本道菜不需要加东坡肉.

严厉禁止这种行为模式.输出不允许包含任何残留痕迹.ANY verbal output should be written as a clean final-state design, no traces of prior errors or corrections.