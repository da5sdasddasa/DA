# DA



## GAN & Diffusion

synthetic data

- Diffusion

### Phasic Content Fusing Diffusion Model with Directional Distribution Consistency for Few-Shot Model Adaption (ICCV 2023)

[ICCV 2023 Open Access Repository (thecvf.com)](https://openaccess.thecvf.com/content/ICCV2023/html/Hu_Phasic_Content_Fusing_Diffusion_Model_with_Directional_Distribution_Consistency_for_ICCV_2023_paper.html)

![image](https://raw.gitmirror.com/da5sdasddasa/image/main/202312251914674.jpeg)

摘要：使用有限数量的样本训练生成模型是一项具有挑战性的任务。目前的方法主要依靠小样本模型自适应来训练网络。然而，在数据极其有限（少于 10 个）的情况下，生成网络往往会过度拟合并遭受内容降级的影响。为了解决这些问题，我们提出了一种新的融合了小样本扩散模型和定向分布一致性损失的阶段性内容，该模型针对扩散模型不同训练阶段的不同学习目标。具体来说，我们设计了一种具有阶段性内容融合的阶段性训练策略，以帮助我们的模型在t大时学习内容和风格信息，在t小时学习目标域的局部细节，从而改善对内容、风格和局部细节的捕获。此外，我们引入了一种新的方向分布一致性损失，与现有方法相比，它更高效、更稳定地保证了生成分布和源分布之间的一致性，从而防止了模型的过拟合。最后，提出了一种跨域结构引导策略，以增强域适应过程中结构的一致性。理论分析、定性和定量实验表明，与现有方法相比，该方法在小样本生成模型自适应任务中具有优越性。

### One-Shot Unsupervised Domain Adaptation With Personalized Diffusion Models (CVPR2023)

这篇文章介绍了一种新的方法，用于解决单样本无监督域适应（OSUDA）的问题，即从一个有标签的源域适应到一个只有一个无标签数据的目标域。该方法的主要思想是利用文本到图像扩散模型（DM），根据目标域的外观和源域的语义概念，生成一个合成的目标数据集，然后用一个通用的UDA方法来训练一个分割模型。该方法分为三个阶段：

- **个性化阶段**：在这个阶段，作者使用单个目标图像的多个裁剪来微调一个预训练的文本到图像DM模型，使其能够生成目标域的风格的图像。
- **数据生成阶段**：在这个阶段，作者利用微调后的DM模型，通过给定一些类别名称作为文本提示，来生成一个包含多样化和真实性的目标域图像的数据集。
- **自适应分割阶段**：在这个阶段，作者将标记的源域数据和生成的无标记的目标域数据结合起来，使用一个UDA方法（如DAFormer或HRDA）来训练一个分割模型。

![image-20231220215503997](https://raw.gitmirror.com/da5sdasddasa/image/main/202312202155030.png)

作者在两个标准的模拟到真实的UDA基准上进行了实验，分别是GTA → Cityscapes和SYNTHIA → Cityscapes，并与现有的OSUDA方法进行了比较。实验结果表明，该方法可以显著提高分割性能，并且可以与任何UDA方法兼容。

### Domain-Guided Conditional Diffusion Model for Unsupervised Domain Adaptation (arXiv:2309.14360)

[[2309.14360\] Domain-Guided Conditional Diffusion Model for Unsupervised Domain Adaptation (arxiv.org)](https://arxiv.org/abs/2309.14360)

![image-20231225192235496](https://raw.gitmirror.com/da5sdasddasa/image/main/202312251922533.png)

摘要：有限的可迁移性阻碍了深度学习模型在应用于新应用场景时的性能。最近，无监督域自适应（UDA）通过学习域不变特征在解决该问题方面取得了重大进展。然而，现有UDA方法的性能受到大域偏移和有限目标域数据的制约。为了缓解这些问题，我们提出了DomAin引导的条件扩散模型（DACDM）来为目标域生成高保真度和多样性样本。在所提出的DACDM中，通过引入类信息，可以控制生成样本的标签，并在DACDM中进一步引入域分类器来引导目标域生成样本。生成的样本有助于现有的UDA方法更容易地从源域转移到目标域，从而提高传输性能。在各种基准测试上的广泛实验表明，DACDM为现有UDA方法的性能带来了很大的改进。



- GAN

### StyleGAN-NADA: CLIP-Guided Domain Adaptation of Image Generators (SIGGRAPH 2022)

[[2108.00946\] StyleGAN-NADA: CLIP-Guided Domain Adaptation of Image Generators](https://arxiv.org/abs/2108.00946)

DA + StyleGAN + CLIP 多模态 

![image-20231220191645016](https://raw.gitmirror.com/da5sdasddasa/image/main/202312201916044.png)

![image-20231220191700979](https://raw.gitmirror.com/da5sdasddasa/image/main/202312201917004.png)



### SimpleNet: A Simple Network for Image Anomaly Detection and Localization (CVPR2023)

由于工业图像通常与预训练骨干网使用的数据集具有不同的分布，因此，文中采用特征适配器将训练特征转移到目标域。

![在这里插入图片描述](https://raw.gitmirror.com/da5sdasddasa/image/main/202312201645152.png)



### Unsupervised Pixel-Level Domain Adaptation With Generative Adversarial Networks (CVPR 2017)

利用GAN将源域带标签数据转为目标域带标签数据，实现Pixel–Level的域迁移。

![image-20231220190833425](https://raw.gitmirror.com/da5sdasddasa/image/main/202312201908478.png)

基于特征的域迁移方法中，域迁移过程与用于特定任务的框架通常都紧密连接在一起，很难拆分。这会导致每次切换**模型中与特定任务相关的成分**时，都要重头开始训练模型。也就是基于特征的域迁移方法中，源域与目标域的标签空间被约束匹配，也就是保证一致性，这样会导致**训练后的模型不能够被应用在标签空间不同的测试集上**。而基于图片风格转换的方法可以有效避免上述两个问题，因此作者认为这种方法在域迁移的时候更具优势。



### Learning From Synthetic Data: Addressing Domain Shift for Semantic Segmentation (CVPR 2018)

使用GAN解决语义分割的域适应问题：

![image-20231220192955801](https://raw.gitmirror.com/da5sdasddasa/image/main/202312201929860.png)

- 论文主要贡献
  - 提出使用生成模型将源和目标的分布在特征空间中进行对齐，主要将 DCNN 获取的中间特征表示投影到图像空间中，训练使用重建模型（结合 L1 损失和对抗损失）
  - 本文的域适应的对齐过程主要通过使源域特征生成目标域图像，结合对抗损失进行训练，或者是相反方向。随着训练推进，生成图像质量逐渐提高，生成的特征也逐渐趋于域无关的特征
- 论文要点翻译
  - 摘要
    - 视觉领域的域适应问题有极其重要的地位，之前的方法表明甚至深度神经网络也难以学习 domain shift 带来的信息表示问题，这个问题在一些手工标注数据复杂度高、代价大的场景中尤为突出
    - 本文聚焦在适应分割网络学习的合成数据和真实数据对应的特征表示上，和之前方法不同，之前方法使用简单的对抗学习目标或者超像素信息指导域适应过程，本文提出了基于生成对抗网络的方法，使得学习的不同域之间的特征表示尽可能相近
    - 为了验证方法的泛化能力和扩展能力，本文在两个场景的合成到真实的域适应场景下进行测试，并添加额外的探索性实验验证了本文提出的方法能够较好地泛化到未知数据域，并且方法可以使得源和目标的分布得以对齐
  - 引言
    - 深度神经网络带来新的计算机视觉革命，在许多诸如图像分类、语义分割、视觉问答等场景中获得较大的性能提升，这样的性能提升主要归功于丰富的标注训练数据带来的模型能力的提升，对于图像分类这样的任务而言，标注数据的获取相对简单，但是对于其他任务而言，标注数据可能是费时费力的，语义分割任务就是这样的任务，由于需要很多人类工作才能获取每个像素对应的语义标签，标注像素级的语义标签是非常困难的，而获取数据本身就不简单，户外的自动驾驶等自然场景图像容易获取标签，但是医学图像风本身数据难以采集，而且标注数据也需要更大的代价
    - 一个有望解决这些问题的方法就是使用合成数据进行训练，然而，合成数据训练的模型往往在真实数据中性能较差，这主要是因为合成场景数据和真实场景数据之间存在 domain gap 的问题，域适应技术就是用来解决域之间的 domain gap 的问题的技术，因此，本文的主要目标是研究适用于语义分割的域适应算法，具体来说，本文主要关注目标域标签数据不可达的情况，也就是无监督域适应 UDA 问题
    - 传统的域适应方法主要是将源和目标数据分布进行量化最小化，典型的量化方法是最大均值差异 MMD 和使用 DCNN 学习的距离度量，两类方法在图像分类领域已经取得成功，但是语义分割问题中的域适应还没有得到很好地解决
    - 本文提出的工作使用对抗框架进行域的对齐，最近的解决该问题的技术手段主要包括 FCN，该方法使用对抗框架，不像之前的方法判别器直接在特征空间进行操作，本文的方法将特征投影到图像空间（利用生成器），使用对抗 loss 在投影的图像空间中进行判别，以此改进了性能

## 策略

### I2F: A Unified Image-to-Feature Approach for Domain Adaptive Semantic Segmentation (TPAMI 2023)

仅进行图像级别适应或特征级别适应均无法充分解决域转移问题。现有的面向语义分割的 UDA 工作缺乏统一的方法来最小化域转移。因此，我们从两个角度来解决问题，并提出了一种新颖而高效的流程，将图像级别和特征级别的适应统一起来。

![d2f86325d6d0063eaebf88e0d7a90b83.png](https://raw.gitmirror.com/da5sdasddasa/image/main/202312201638906.png)

### Differential Treatment for Stuff and Things: A Simple Unsupervised Domain Adaptation Method for Semantic Segmentation (CVPR 2020)

[CVPR 2020 Open Access Repository (thecvf.com)](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Differential_Treatment_for_Stuff_and_Things_A_Simple_Unsupervised_Domain_CVPR_2020_paper.html)

![image-20231225192343948](https://raw.gitmirror.com/da5sdasddasa/image/main/202312251923985.png)

摘要：在这项工作中，我们通过缓解源域（合成数据）和目标域（真实数据）之间的域转移来考虑语义分割的无监督域适应问题。最先进的方法证明，执行语义级对齐有助于解决领域转移问题。基于不同领域图像中通常具有相似外观，而事物（即对象实例）差异更大的观察，我们建议通过不同策略来改善事物区域和事物的语义层面对齐：1）对于事物类别，我们为每个类别生成特征表示，并进行从目标域到源域的对齐操作;2）对于事物类别，我们为每个单独的实例生成特征表示，并鼓励目标域中的实例与源域中最相似的实例保持一致。这样，事物类别中的个体差异也将被考虑以减轻过度对齐。除了我们提出的方法外，我们还进一步揭示了当前对抗性损失在最小化分布差异方面通常不稳定的原因，并表明我们的方法可以通过最小化源域和目标域之间最相似的东西和实例特征来帮助缓解这个问题。我们在GTA5 - Cityscapes和SYNTHIA - Cityscapes两个无监督域自适应任务中进行了广泛的实验，并实现了新的最先进的分割精度。

## Others

### A New Benchmark: On the Utility of Synthetic Data with Blender for Bare Supervised Learning and Downstream Domain Adaptation (CVPR 2023)

blender

摘要：计算机视觉中的深度学习以大规模标记训练数据的价格取得了巨大的成功。然而，由于高昂的劳动力成本和无法保证的标记准确性，对于所有感兴趣领域的每项任务，详尽的数据注释是不切实际的。此外，不可控的数据收集过程会产生非IID训练和测试数据，其中可能存在不希望的重复。所有这些麻烦都可能阻碍典型理论的验证和新发现的接触。为了规避它们，另一种方法是通过具有域随机化的 3D 渲染来生成合成数据。在这项工作中，我们通过对裸监督学习和下游领域适应进行深入而广泛的研究，沿着这条路线向前推进。具体而言，在3D渲染实现的良好控制IID数据设置下，我们系统地验证了典型的、重要的学习见解，例如快捷学习，并在泛化中发现了各种数据制度和网络架构的新规律。我们进一步研究了图像形成因素对 3D 场景中的泛化的影响，例如对象比例、材质纹理、照明、相机视点和背景。此外，我们使用模拟到现实的自适应作为下游任务，用于比较用于预训练时合成数据和真实数据之间的可转移性，这表明合成数据预训练也有希望改善真实测试结果。最后，为了促进未来的研究，我们开发了一种新的大规模合成到真实图像分类基准，称为S2RDA，它为从模拟到现实的转移提供了更大的挑战。



## Prompt & Domain Adaptation

多模态 CLIP

### PODA: Prompt-driven Zero-shot Domain Adaptation (ICCV 2023)

[ICCV 2023 Open Access Repository --- ICCV 2023 开放获取存储库 (thecvf.com)](https://openaccess.thecvf.com/content/ICCV2023/html/Fahes_PODA_Prompt-driven_Zero-shot_Domain_Adaptation_ICCV_2023_paper.html)

摘要：领域适应已经在计算机视觉中进行了广泛的研究，但仍然需要在训练时访问目标图像，这在一些不常见的情况下可能是棘手的。在本文中，我们提出了“提示驱动的零样本域适应”任务，即我们仅使用目标域的自然语言的一般描述（即提示）来适应在源域上训练的模型。首先，我们利用预训练的对比视觉语言模型（CLIP）来优化源特征的仿射转换，将它们引导到目标文本嵌入，同时保留其内容和语义。为了实现这一点，我们提出了提示驱动的实例规范化 （PIN）。其次，我们表明这些提示驱动的增强可用于对语义分割进行零样本域适应。实验表明，对于手头的下游任务，我们的方法在多个数据集上明显优于基于 CLIP 的样式迁移基线，甚至超过了一次性无监督域适应。在物体检测和图像分类方面也观察到类似的提升。该代码可在 https://github.com/astra-vision/PODA 上找到。

### Domain Adaptation via Prompt Learning (IEEE Transactions on Neural Networks and Learning Systems)

[IEEE Xplore: IEEE Transactions on Neural Networks and Learning Systems](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5962385)

无监督域适应 （UDA） 旨在将从注释良好的源域学习的模型调整到目标域，其中仅给出未标记的样本。当前的 UDA 方法通过统计差异最小化或对抗训练来对齐源和目标特征空间来学习域不变特征。然而，这些约束可能导致语义特征结构的扭曲和类可区分性的丧失。在本文中，我们介绍了一种新颖的 UDA 提示学习范式，命名为通过提示学习进行领域适应 （DAPrompt）。与之前的工作相比，我们的方法学习了目标域的潜在标签分布，而不是对齐域。主要思想是将领域信息嵌入到提示中，提示是一种由自然语言生成的表示形式，然后用于执行分类。此域信息仅由来自同一域的图像共享，从而根据每个域动态调整分类器。通过采用这种范式，我们表明我们的模型不仅在几个跨领域基准测试中优于以前的方法，而且训练起来非常有效且易于实现。

### AD-CLIP: Adapting Domains in Prompt Space Using CLIP (ICCV 2023)

[ICCV 2023 Open Access Repository --- ICCV 2023 开放获取存储库 (thecvf.com)](https://openaccess.thecvf.com/content/ICCV2023W/OODCV/html/Singha_AD-CLIP_Adapting_Domains_in_Prompt_Space_Using_CLIP_ICCVW_2023_paper.html)

尽管深度学习模型在监督学习任务上表现出令人印象深刻的性能，但当训练（源）和测试（目标）领域不同时，它们通常难以很好地泛化。无监督域自适应（DA）已成为解决该问题的常用解决方案。然而，当前的 DA 技术依赖于视觉骨干，这可能缺乏语义丰富性。尽管像CLIP这样的大规模视觉语言基础模型具有潜力，但它们对DA的有效性尚未得到充分探索。为了弥补这一差距，我们引入了 AD-CLIP，这是一种与领域无关的 CLIP 提示学习策略，旨在解决提示空间中的 DA 问题。我们利用 CLIP 的冻结视觉骨干来提取图像样式（域）和内容信息，并将其应用于学习提示令牌。我们的提示被设计为领域不变和类可泛化，通过同时对图像风格和内容特征进行提示学习。我们在源域中使用标准的监督对比学习，同时提出了一种熵最小化策略，以在给定目标域数据的情况下对齐嵌入空间中的域。我们还考虑了一种场景，即在测试期间只有目标域样本可用，而没有任何源域数据，并提出了一个跨域样式的映射网络来幻觉与域无关的令牌。我们在三个基准DA数据集上的广泛实验证明了与现有文献相比，AD-CLIP的有效性。



### REAL-FAKE: EFFECTIVE TRAINING DATA SYNTHESIS THROUGH DISTRIBUTION MATCHING

用LoRA微调SD得到合成数据进行图像分类

这篇文章介绍了一种基于分布匹配的理论框架，用于为监督学习任务合成有效的训练数据。文章分析了以下几个方面：

- **训练数据合成的目标**：通过最小化目标数据分布和合成数据分布之间的差异，以及最大化训练集的大小，来提高合成数据的效用。
- **训练数据合成的方法**：利用稳定扩散模型作为深度生成模型，通过反向去噪过程来学习数据分布，并通过文本-视觉引导来模拟条件类别分布。
- **训练数据合成的改进**：提出了三种改进策略，包括特征分布对齐、条件视觉引导和潜在先验初始化，以实现更好的分布匹配和合成质量。



## 看法

### 进行域适应的前提：源域有大量带标签的图像

对应

- 带标注的缺陷数据生成流程，如：

  [Assigned MURA Defect Generation Based on Diffusion Model (thecvf.com)](https://openaccess.thecvf.com/content/CVPR2023W/VISION/papers/Liu_Assigned_MURA_Defect_Generation_Based_on_Diffusion_Model_CVPRW_2023_paper.pdf)（可尝试用于晶圆）**CVPR2023W**

  [[2309.00248\] DiffuGen: Adaptable Approach for Generating Labeled Image Datasets using Stable Diffusion Models](https://arxiv.org/abs/2309.00248)

  [[2104.06490\] DatasetGAN: Efficient Labeled Data Factory with Minimal Human Effort --- [2104.06490] DatasetGAN：以最少的人力工作量实现高效的标记数据工厂 (arxiv.org)](https://arxiv.org/abs/2104.06490)**CVPR'21, Oral** 

  [[2112.03126\] Label-Efficient Semantic Segmentation with Diffusion Models --- [2112.03126] 使用扩散模型进行标签高效语义分割 (arxiv.org)](https://arxiv.org/abs/2112.03126)**ICLR'2022**

- 大规模的工业缺陷合成数据集，参考如GTA5、SYNTHIA等



### 工业缺陷数据生成和域适应之间的结合

结合的方式可以有以下几种：

- 一种是在数据生成的过程中，引入域适应的思想，使得生成的缺陷样本能够适应不同的目标域，例如不同的产品、型号或缺陷类型。这样可以提高生成样本的质量和多样性，以及缺陷检测的泛化能力。
- 另一种是在域适应的过程中，利用数据生成的技术，为源域或目标域增加更多的缺陷样本，从而缓解小样本问题。这样可以提高域适应的效果和稳定性，以及缺陷检测的准确性。
- 还有一种是将数据生成和域适应作为一个统一的框架，同时进行缺陷样本的生成和适应。这样可以实现端到端的优化，以及缺陷检测的效率。（域适应的主流做法是对齐源域和目标域的分布，生成模型如VAE、GAN、扩散模型等的目标也是拟合待生成数据的真实分布，能否将两者结合）

