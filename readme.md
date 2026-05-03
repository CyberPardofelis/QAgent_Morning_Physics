[![GitHub release (latest by date)](https://img.shields.io/github/v/release/CyberPardofelis/QAgent_Morning_Physics)](https://github.com/CyberPardofelis/QAgent_Morning_Physics/releases/latest)
[![Downloads](https://img.shields.io/github/downloads/CyberPardofelis/QAgent_Morning_Physics/total)](https://github.com/CyberPardofelis/QAgent_Morning_Physics/releases)

## **Quantum-Physics Agent**

### 预编译版本
从[GitHub Releases](https://github.com/CyberPardofelis/QAgent_Morning_Physics/releases/latest) 页面下载最新的文件压缩包，解压后运行`light.exe`

发送 早上好 + 名称以获取新文章（或近期文章）总结，如 早上好 arxiv，早上好 prl 等\
可以继续展开其中某一个领域总结所有文章（目前只针对arxiv）\
期刊目前仅支持APS的部分期刊，如PRA/B/C/D/E/L/X,PRX Quantum, Rev. Mod. Phys.\
对感兴趣的论文可以下载并完整分析，生成格式化文章分析结果并保存\
对文章有评论也可以保存至分析结果\
目前的问题：依靠DOI的APS文章下载方案是基于playwright的，如果觉得慢可以按题目搜索，会自动使用arxiv搜索和下载工具。

**注意** 运行需要配置API密钥\
分区选择在设置界面\
密钥需要写入.env文件，格式如下：\
DEEPSEEK_API_KEY=\
OPENAI_API_KEY=\
本项目默认使用deepseek，调试也是在deepseek上调试的。\
虽然加入了Open AI功能，但并没有用过。原则上可以，实际上难说。\
保存文件后重新启动程序即可

更新了APS的网页格式适配，这APS真会给人整活\
完成了工具函数和UI相关函数的文件拆分，便于后续更新\
更新了LLM调用的相关函数，留出参数输入接口，便于适配DS的新模型等等。

后续会加入更多平台和更多分区的兼容。\
但现在还没。反正饼先画上，慢慢做\
当然如果能全部支持的基础上保留默认并根据对话选择分区并且不同平台统一就更好了，但是我还没想好要怎么做（逃\
对于更多平台，RSS可以作为首选项，毕竟确实方便\
但是某些平台（APS）的RSS说得就像说了一样，不能说没有用但我想拿到更丰富的信息\
因此有了采取了控制浏览器打开网页抓取html信息的邪修作为备选\
反正RSS也可能格式、字段名什么的不一样，和针对平台的html结构分析信息也未必有本质区别\
更何况不管怎么做都是压力deepseek，其实对我来说没区别（逃

它凝结了无差别的AI劳动\
以及无差别的人类昏厥\
还有无差别的汽水薯条
