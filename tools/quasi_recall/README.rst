.. role:: raw-html-m2r(raw)
   :format: html


CF指标工具
==========

Demo可选择两个模式执行:
1) 通用模式: 提供给公司外部使用\ :raw-html-m2r:`<br>`
2) 自定义模式: 提供给百度内部根据策略不同需求可自行开发使用\ :raw-html-m2r:`<br>`
其中demo中提供了以手百图文ucf策略的评估(插件命名为shoubai)，其他策略可自定义实现  

【准备工作】\ :raw-html-m2r:`<br>`
在执行程序之前，先按照如下说明进行配置准备：\ :raw-html-m2r:`<br>`
一、config.yaml 配置说明如下：\ :raw-html-m2r:`<br>`
1、ugi:  模型产出向量存放afs的ugi账号信息（格式：用户名, 密码）\ :raw-html-m2r:`<br>`
2、afs_conf:  指定本地执行hadoop对应的hadoop-site.xml文件 （前提需安装hadoop)\ :raw-html-m2r:`<br>`
3、base_user_data/new_user_data:  (用户向量地址，可支持afs或者ftp)\ :raw-html-m2r:`<br>`
如果有两组实验对比（只支持最多两组实验的对比): 分别填写对应向量地址;\ :raw-html-m2r:`<br>`
如果只需评估一组实验: 只需在new_user_data填写\ :raw-html-m2r:`<br>`
4、data_type: 评估资源类型(图文 or 视频)\ :raw-html-m2r:`<br>`
5、case_data: 评估用例地址(可选填，如果不填写，则从向量中获取评估用例)\ :raw-html-m2r:`<br>`
以ucf策略为例，
如果是两组实验评估，此处不用填，会在两组实验的向量中取公共uids并设置可达最大阈值为5000作为评估用例
如果是一组实验，则从用户向量文件中选取上限为5000的评估用例\ :raw-html-m2r:`<br>`
6、max_records:  用于下载向量数据库控制评估向量条数（目前2kw+条数据，评估计算耗时3h左右； 如果是100w+条数据，评估计算耗时约10min左右）\ :raw-html-m2r:`<br>`
7、user_time: 用于自定义模式定义评估ucf的时间阈值，格式如1598760000\ :raw-html-m2r:`<br>`
8、item_time: 用于自定义模式定义评估icf的时间阈值，格式如"2020-08-30 00:00:00"\ :raw-html-m2r:`<br>`
9、plan_type:  评估策略类型（目前demo支持ucf, 其余可自行定义）\ :raw-html-m2r:`<br>`
10、dim: 向量维度数量\ :raw-html-m2r:`<br>`
11、n_trees & search_n: annoy计算使用\ :raw-html-m2r:`<br>`
12、plugins_type:  

.. code-block::

   * 填写"shoubai" (自定义模式): 以手百图文ucf策略评估
   * 填写"common" (通用模式): 无需填写上述1～11点  

13、jobname:  准召评估任务的名称定义（不同任务定义对应的名称，后续会将对应的数据以定义的名称作为目录存储，方便问题排查及定位）  

二、如果需执行demo的shoubai插件, 需先额外的so文件\ :raw-html-m2r:`<br>`
【执行命令】\ :raw-html-m2r:`<br>`
注意: 通用模式和自定义模式区别在于config.yaml配置中"plugins_type"区别使用\ :raw-html-m2r:`<br>`
1) 通用模式: 准备好上述一条件后，执行譬如:\ :raw-html-m2r:`<br>`
python run.py --truth '{"12345": ["23455", "123455"], "234": ["23422"]}' --recall '{"12345": ["23455", "123455"], "234": ["23422"]}'

2) 自定义模式: 上述三点均准备就绪后，按照如下运行即可:\ :raw-html-m2r:`<br>`
python run.py  
