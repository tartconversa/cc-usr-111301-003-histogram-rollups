# 异构延迟直方图的可证精确汇总

本项目保存虚构算力设施的仓库源表、湖文件与Python读取工具。现有工具加载输入并输出可证精确或明确标注为可行域的汇总结果，不做单次请求级加工，也不接触请求正文、模型响应、轨迹或图片。所有设备、组织和金额均为合成。

## 字段与单位

每叶histogram列edges与counts，最后边界字符串+inf表示无穷尾桶。`null_count`单独保存，**不进入分位分母**。`histogram_id`/`revision`是替换身份：同一id的高revision撤销旧贡献再加入新贡献，不是可重复相加的标签。

material-index.json记录全部输入内容哈希；fixtures保存原始资料与配置，计算输出可放.workspace-output，原始资料不应被覆盖。

## 汇总层契约

* **精确重分桶**：目标边界必须与源边界重合、落在支撑下界之前，或落在**零计数**桶内部。切开非空桶时抛 `BoundsError`，绝不插值后仍称精确。
* **自动汇总**取所有来源共同边界构成的最细公共粗化（各叶有限边界交集 + 全局最小下界），再逐桶相加；`+inf` 尾桶只在来源声明或尾段实际承载计数时保留。
* **手动细边界**若无法精确切分，返回每桶联合可行计数区间、精确守恒总量与一组见证分配；逐桶下界之和与上界之和通常偏离总量，这正是“独立上下界相加制造漂移”的反例。
* **分位**采用最小累计达到 `ceil(p*N)` 的 1 基秩定义，报告命中桶的半开值域；`+inf` 上界保持无穷，空总体标为 `defined: false`。各叶 P99 不会被平均成总体 P99。
* **父层版本**由叶版本集合 `(id, revision, digest)` 决定；同一叶来源不得从两个父入口重复进入。合并树记录撤销/新增事件，可持久化回读。

## 运行

Python 3.11以上，标准库即可，不需安装第三方依赖。

    python -m src.histogram_reader fixtures/histograms.json
    python -m src.histogram_rollup fixtures/histograms.json fixtures/rollups.json
    python -m src.histogram_rollup fixtures/histograms.json fixtures/rollups.json --json
    python -m compileall -q src tests
    python -m unittest discover -s tests -v

fixtures 示例：源A桶[0,10)/[10,20)计2/2，源B桶[0,5)/[5,20)计1/3；精确公共粗化只有[0,20)计8。请求目标10会切开B的非空桶，故只给出可行域。A单独50%分位秩2位于[0,10)。

## 固定容器

    docker build -t histogram-rollups:local .
    docker run --rm histogram-rollups:local

容器镜像以完整digest固定，默认仅检查输入契约；实现新增分析功能所需依赖由开发者明确登记。本汇总层只依赖标准库。
