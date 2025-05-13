# MOF-Architect (TopoSpace)

## Overview

In this work, we develop an efficient data-driven machine learning workflow for predicting metal node types of metal-organic framework (MOF) materials. Models trained according to this workflow were able to instantly predict the metal node type (Paddle-wheel, Rod, or Other) of Cu-carboxylate MOFs, achieving excellent prediction accuracy (91%), precision (89%), and recall (85%) utilizing the extreme gradient boosting (XGBoost) algorithm. Using this powerful prediction model, we designed two tricarboxylic acid ligands functionalized by different sterically hindered molecules based on the predictions of the model output, and demonstrated the successful prediction of Cu-SBUs types by synthesis experiments (the model successfully guided the directed synthesis of two novel functionalized MOFs [SJTU-2 and SJTU-5]). This scheme breaks through the limitations of the traditional trial-and-error method and provides an AI-driven solution for the rational design of MOFs.

## Core Workflows (using Cu-carboxylate MOFs as an example)

### 1. Data Extraction and Feature Engineering
- ​**Dataset Construction**：The MOF dataset utilized in this study was sourced from the Cambridge Crystallographic Data Centre (CCDC) database (CSD Version 5.45). To construct an experimental dataset specifically for Cu-MOFs, MOFs where copper (Cu) is exclusively coordinated with carboxylate groups were selected as the filtering criteria. A total of 566 Cu-MOF structures meeting these criteria were retrieved from the CSD. Subsequently, the metal nodes in these Cu-MOFs were classified based on their topological characteristics into three types: "Paddle-wheel" (458 entries), "Rod" (56 entries), and "Other" (52 entries). This classification ensured that the structural types of metal nodes were explicitly annotated within the dataset.
- ​**Organic Linker Information Extraction and Feature Vector Integration**：During the construction of "feature vectors," we extract multi-dimensional information from the organic linkers, including SMILES representations, chemical descriptors, and molecular fingerprints, to comprehensively capture their physicochemical properties. Specifically, these features include fundamental molecular attributes (e.g., ring structures, chain structures, and functional groups), topological properties (e.g., polar surface area), electronic properties (e.g., minimum and maximum partial charges), and other structural parameters (e.g., the number of rotatable bonds and hydrogen bond donors/acceptors). These properties are integrated into a high-dimensional feature vector through feature engineering, representing the complex molecular structure of organic linkers.

### 2. Machine Learning Modeling
- ​**Algorithm Selection**：In this work, we conducted a comprehensive performance evaluation of several commonly used machine learning models, including Support Vector Machines (SVM), Random Forest (RF), and Extreme Gradient Boosting (XGBoost), among others. Based on prediction accuracy, scalability, and computational efficiency, XGBoost was ultimately chosen as the core model for this workflow. XGBoost, a gradient boosting framework based on decision trees, is widely employed due to its exceptional performance in handling high-dimensional features and capturing nonlinear relationships.
- ​**Model Optimization**：
  - Systematic hyperparameter tuning for the XGBoost model was conducted using Grid Search.
  - Three-fold cross-validation was employed during the grid search process to evaluate the robustness of the model. The model with the highest cross-validation score was selected as the optimal model.

### 3. Prediction-Experiment Closed-loop Validation
- ​**Linker Design**：Based on the previously reported Cu-MOF (MOF-14) comprising of C3-symmetric tritopic linker and paddle-wheel Cu nodes, we specifically designed two similar linkers by introducing additional three methyl (H3BTB-Me) or 4’-tert-butyl phenyl (H3BTB-t-Bu-phenyl) spacers into the central benzene ring while retaining the 3-fold symmetry.
- ​**Model Prediction**：
  - H3BTB-Me → Paddle-wheel node
  - H3BTB-t-Bu-phenyl → Rod node
- ​**Experimental Validation**：
  - SJTU-2（Paddle-wheel）：novel interdigitated 3D architecture formed through inclined interpenetration of 2D layers
  - SJTU-5（Rod）：a permanent ultramicroporous material featuring helical 1D channels with highly inert surfaces root from abundant tert-butyl groups, which results in preferential adsorption of propane (C3H8) over propene (C3H6)

## Open Source License
This project utilizes the **MIT License**：
- Allow any user to freely use, modify, and distribute the code and documentation of this project, provided that the copyright notice and license documents are retained.
- Users can develop derivative projects based on this framework (including all types of uses).

*Note: See LICENSE document for full agreement details.*

Now you can make your initial attempts on this site: https://mof-architect-testforpublic-production.up.railway.app/mof-architect
For more information on TopoSpace, see: https://topo-space.com/ (website under construction)

# MOF-Architect （拓扑空间）

## Overview

在这项工作中，我们开发了一种高效的数据驱动机器学习工作流，用于预测金属-有机框架（MOF）材料的金属节点类型。按照此工作流训练的模型能够即时预测铜-羧酸盐 MOFs 的金属节点类型（Paddle-wheel、Rod或其他），利用极端梯度提升 (XGBoost) 算法实现了极佳的预测准确率（91%）、精确率（89%）和召回率（85%）。利用这一强大的预测模型，我们基于模型输出的预测结果设计了两种由不同立体受阻分子功能化的三位羧酸配体，并通过合成实验证明了 Cu-SBUs 类型的成功预测（模型成功指导了两种新型功能化MOF[SJTU-2和SJTU-5]的定向合成）。本方案突破了传统试错法局限，为MOF的理性设计提供了AI驱动的解决方案。

## 核心工作流（以铜-羧酸盐 MOFs为例）

### 1. 数据构建与特征工程
- ​**数据集构建**：本研究所使用的MOF数据集来源于CCDC数据库（CSD Version 5.45）。为构建针对Cu-MOF的实验数据集，以金属Cu仅与羧酸基团配位的MOF为筛选条件，从CSD中检索并收集了566种符合条件的Cu-MOF结构。进一步根据拓扑特征对这些Cu-MOF的金属节点进行分类，将其分为“Paddle-wheel”（458个）、“Rod”（56个）和“Other”（52个）三种类型，以确保金属节点的结构类型在数据集中的明确标注。
- ​**有机配体多维特征提取**：
在“特征向量”的构建过程中，通过提取有机配体的SMILES表示、化学描述符以及分子指纹等多维信息，全面捕捉配体的物理化学特性。具体而言，这些特征包括有机配体的基本分子属性（例如环状结构、链状结构和官能团等）、拓扑性质（例如极性表面积）、电荷性质（例如最小和最大部分电荷），以及其他结构性参数（例如可旋转键数量和氢键供体/受体数量）。这些信息通过特征工程被整合为一个高维度的特征向量，充分表征了有机配体的复杂分子结构。

### 2. 机器学习建模
- ​**算法选择**：在本研究中，我们对多种常用的机器学习模型（例如支持向量机（SVM）、随机森林（Random Forest）和Extreme Gradient Boosting（XGBoost）等）进行了全面的性能评估。基于预测准确性、扩展性及计算效率，最终选择了XGBoost作为我们的核心模型算法。XGBoost是一种基于决策树的梯度提升框架，因其在处理高维特征和非线性关系方面的优异性能而被广泛应用。
- ​**模型优化**：
  - 使用网格搜索（Grid Search）对XGBoost模型的超参数进行系统化调优
  - 在网格搜索过程中使用三折交叉验证评估模型的鲁棒性。最终选择交叉验证得分最高的模型作为最佳模型。

### 3. 预测-实验闭环验证
- ​**配体设计**：对原型配体H3BTB进行甲基/叔丁基苯功能化改造
- ​**模型预测**：
  - H3BTB-Me → Paddle-wheel节点
  - H3BTB-t-Bu-phenyl → Rod节点
- ​**实验验证**：
  - SJTU-2（Paddle-wheel）：层间倾斜互穿三维结构
  - SJTU-5（Rod）：螺旋超微孔道&丙烷/丙烯逆向分离

## 开源协议
本项目采用**MIT开源协议**：
- 允许任何用户在保留版权声明及许可文件的前提下，自由使用、修改、分发本项目的代码与文档
- 使用者可基于本框架开发衍生项目（包括各类用途）

*注：完整协议细节详见LICENSE文件*

现在你可以在这个网站上进行初步的尝试: https://mof-architect-testforpublic-production.up.railway.app/mof-architect
有关TopoSpace的更多信息，请见：https://topo-space.com/（网站搭建中）
