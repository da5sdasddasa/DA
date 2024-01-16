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



结合多模态：

2D与3D结合 nerf等

文本、视频等？

**文本数据：** 将相关文本信息（例如设备说明书、维护记录、报告）与图像数据结合，可以提供更多的背景信息，有助于更好地理解和分析缺陷。自然语言处理（NLP）技术可以用于处理和分析这些文本信息。

标签生成、controlnet



## 多模态数据生成

### ImageBind: One Embedding Space To Bind Them All (CVPR 2023)

[ImageBind: One Embedding Space To Bind Them All (thecvf.com)](https://openaccess.thecvf.com/content/CVPR2023/papers/Girdhar_ImageBind_One_Embedding_Space_To_Bind_Them_All_CVPR_2023_paper.pdf)

[IMAGEBIND：统一六种模态的嵌入空间 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/629386507)

![v2-68a0f8682587eaf3f8eaab43dc502f7c_b](https://raw.gitmirror.com/da5sdasddasa/image/main/202401031915657.gif)

将文本、深度、IMU、音频和热图通过图像绑定在一起，对齐多种模态，可以重点关注的点：

无需重新训练，我们可以 "升级 "现有的使用CLIP嵌入的视觉模型，以使用来自其他模态的IMAGEBIND嵌入，如音频。将基于文本的检测器升级为基于音频的检测器。我们使用一个预训练好的基于文本的检测模型，Detic [86]，并简单地用IMAGEBIND的音频嵌入替换其基于CLIP的'类'（文本）嵌入。无需训练，这就创造了一个基于音频的检测器，可以根据音频提示来检测和分割目标。如图5所示，我们可以用狗的吠叫声来提示检测器，以定位一只狗。

![image-20240103192249401](https://raw.gitmirror.com/da5sdasddasa/image/main/202401032022519.png)

view: 直接用音频嵌入替换文本嵌入即可将基于文本的检测器升级为基于音频的检测器，能否将这种思路用于多模态数据生成

### Collaborative Diffusion for Multi-Modal Face Generation and Editing (CVPR 2023)

[Collaborative Diffusion for Multi-Modal Face Generation and Editing (thecvf.com)](https://openaccess.thecvf.com/content/CVPR2023/papers/Huang_Collaborative_Diffusion_for_Multi-Modal_Face_Generation_and_Editing_CVPR_2023_paper.pdf)

[CVPR 2023 | 多个扩散模型相互合作，新方法实现多模态人脸生成与编辑 (qq.com)](https://mp.weixin.qq.com/s/UdGFbp6xPRaMGy4L2uOrJw)



CVPR 2023 的 Collaborative Diffusion 提供了一种简单有效的方法来实现不同扩散模型之间的合作。

![image-20240103193052247](https://raw.gitmirror.com/da5sdasddasa/image/main/202401031930320.png)

不同种类的扩散模型性能各异 —— text-to-image 模型可以根据文字生成图片，mask-to-image 模型可以从分割图生成图片，除此之外还有更多种类的扩散模型，例如生成视频、3D、motion 等等。

![图片](https://raw.gitmirror.com/da5sdasddasa/image/main/202401032022089.jpeg)

假如有一种方法让这些 pre-trained 的扩散模型合作起来，发挥各自的专长，那么我们就可以得到一个多功能的生成框架。比如当 text-to-image 模型与 mask-to-image 模型合作时，我们就可以同时接受 text 和 mask 输入，生成与 text 和 mask 一致的图片了。CVPR 2023 的 Collaborative Diffusion 提供了一种简单有效的方法来实现不同扩散模型之间的合作。

view: 用否用类似的方式将不同输入输出模态的生成模型协调结合起来以实现多模态的数据生成



## 可控图像生成

### CVPR 2023 | FreestyleNet：自由式布局到图像生成

[openaccess.thecvf.com/content/CVPR2023/papers/Xue_Freestyle_Layout-to-Image_Synthesis_CVPR_2023_paper.pdf](https://openaccess.thecvf.com/content/CVPR2023/papers/Xue_Freestyle_Layout-to-Image_Synthesis_CVPR_2023_paper.pdf)

[CVPR 2023 | FreestyleNet：自由式布局到图像生成 (qq.com)](https://mp.weixin.qq.com/s/8Xnm4GrrurTln06sc_nNjA)

本文基于预训练的文生图大模型Stable Diffusion 构建了FreestyleNet。Stable Diffusion能够为我们提供丰富的语义，但是其只支持文本作为输入，如何将这些语义填入指定的布局是一个巨大的挑战。为此，本文引入了修正交叉注意力（Rectified Cross-Attention，RCA）层，并将其插入到Stable Diffusion的U-Net当中。通过限制文本token只在特定的区域与图像token产生交互，RCA实现了将语义自由放置在指定布局上的功能。实验表明，FreestyleNet能够结合文本和布局生成逼真的结果，进一步增强了图像生成的可控性。

![image-20240103200018066](https://raw.gitmirror.com/da5sdasddasa/image/main/202401032022147.png)



***



如何在图像生成的同时得到像素级标签即可控生成已经有一些方法 如controlnet freestylenet等 问题：由于扩散过程的随机性，生成的标签仍然不是很可靠，将这种数据用于训练会干扰模型的性能，如何实现标签可靠的数据生成。

### NeurIPS 2023｜FreeMask: 用密集标注的合成图像提升分割模型性能

[[2310.15160\] FreeMask: Synthetic Images with Dense Annotations Make Stronger Segmentation Models --- [2310.15160] FreeMask：具有密集注释的合成图像可以打造更强的分割模型 (arxiv.org)](https://arxiv.org/abs/2310.15160)

[NeurIPS 2023｜FreeMask: 用密集标注的合成图像提升分割模型性能 (qq.com)](https://mp.weixin.qq.com/s/xLiLOjFfFkLA4Davl3u-yQ)

合成数据的策略及如何利用合成数据：

![image-20240103203436537](https://raw.gitmirror.com/da5sdasddasa/image/main/202401032037109.png)



![image-20240103203740297](https://raw.gitmirror.com/da5sdasddasa/image/main/202401032039653.png)

![image-20240103193952289](https://raw.gitmirror.com/da5sdasddasa/image/main/202401032022169.png)



view：文章中的方法用ADE20K和COCO数据集的mask生成一个更大的ADE20K-Synthetic数据集（包含ADE20K的20倍的训练图像）和COCO-Synthetic数据集（包含COCO-Stuff-164K的6倍的训练图像）然后将真实图像与合成图像混合训练，验证合成图像对分割模型性能的提升。需要关注合成图像的标签质量和合成图像的复杂程度(困难样本)





## 合成数据

如果合成数据集和真实数据集之间的差异很大，那么使用域适应的方法可以提高模型在真实数据集上的性能。如果合成数据集和真实数据集之间的差异很小，那么使用域适应的方法可能没有太大的必要或效果。一般来说，如果生成方法能够保证合成数据和真实数据的分布一致性，那么就不需要进行域适应；如果生成方法导致合成数据和真实数据的分布存在较大差异，那么就需要进行域适应，以提高模型的泛化能力。

目前绝大多数的数据生成论文在用于分类、检测等视觉任务上时都没有涉及域适应，原因可能在于数据生成在于拟合真实数据的分布，如果使用域适应方法说明该数据生成方法学习的分布与真实分布差异大，效果不好。



目前的数据生成方法已经十分逼真，合成数据可直接用于视觉任务的精度提升

[Explore the Power of Synthetic Data on Few-shot Object Detection](https://openaccess.thecvf.com/content/CVPR2023W/GCV/html/Lin_Explore_the_Power_of_Synthetic_Data_on_Few-Shot_Object_Detection_CVPRW_2023_paper.html)CVPRW2023:探索合成数据在少样本目标检测中的作用

随着生成模型的发展，文本到图像的生成已经取得了很大的进展。例如，DALLE、Imagen和Stable Diffusion可以通过简单地使用输入的文本描述生成高质量的图像。这些生成器可以产生不同的结果，这意味着工业应用的前景更加光明，例如解决许多现有的少数或长尾问题。这鼓励我们探索文本到图像生成器的合成数据对FSOD任务的影响。我们的研究是在开源的稳定扩散上进行的。通过使用生成的图像，我们定义了一个新的(K + G)-shot设置，用于使用合成数据进行少镜头学习的问题，由K个真实的新实例和G个生成的新实例组成。必须回答两个关键问题:(1)如何将合成数据用于FSOD?(2)如何从大规模合成数据集中找到具有代表性的样本?

[Is synthetic data from generative models ready for image recognition?](https://arxiv.org/abs/2210.07574)用合成数据做训练，效果比真实数据还好丨ICLR 2023 

使用高质量AI合成图片，来提升**图像分类模型**的性能

[Synthetic Data from Diffusion Models Improves ImageNet Classification](https://arxiv.org/pdf/2304.08466.pdf)使用扩散模型生成合成图像，将真实数据集进行了扩充，从而提高了在ImageNet分类任务上的效果。这说明了使用图像生成方法可以增加数据的多样性，提高模型的泛化能力。但是，这并不意味着一定要使用域适应的方法，因为扩散模型生成的图像和真实图像之间的差异并不是很大，而且扩散模型本身就是在真实图像上训练的。





在数据阶段，域适应更适合以下场景：

blender等物理仿真数据、图像合成、光照背景尺度等变化引起的Domain Shift



[Making Images Real Again: A Comprehensive Survey on Deep Image Composition]([2106.14490.pdf (arxiv.org)](https://arxiv.org/pdf/2106.14490.pdf))使用深度学习方法进行图像合成，即将一张图片的前景剪切下来，粘贴到另一张图片的背景上，得到一张新的图片。但是由于前景和背景之间可能存在不一致性，例如颜色、光照、透视等，所以需要使用域适应的方法来消除这些不一致性，让合成图看起来更加真实自然。

[Industrial Anomaly Detection with Domain Shift: A Real-world Dataset and Masked Multi-scale Reconstruction](https://arxiv.org/abs/2304.02216)

工业异常检测 (IAD) 对于自动化工业质量检测至关重要。数据集的多样性是开发综合IAD算法的基础。现有的IAD数据集关注数据类别的多样性，忽视了同一数据类别内域的多样性。在本文中，为了弥补这一差距，我们提出了航空发动机叶片异常检测（AeBAD）数据集，该数据集由两个子数据集组成：单叶片数据集和叶片视频异常检测数据集。与现有数据集相比，AeBAD具有以下两个特点：1.）目标样本未对齐且尺度不同。 2.) 测试集和训练集中正态样本的分布存在域偏移，其中域偏移主要是由光照和视图的变化引起的。基于该数据集，我们观察到，当测试集中正常样本的域发生变化时，当前最先进的 (SOTA) IAD 方法表现出局限性。为了解决这个问题，我们提出了一种称为掩模多尺度重建（MMR）的新方法，它增强了模型通过掩模重建任务推断正常样本中斑块之间因果关系的能力。与 AeBAD 数据集上的 SOTA 方法相比，MMR 实现了卓越的性能。此外，MMR 通过 SOTA 方法实现了具有竞争力的性能，以检测 MVTec AD 数据集上不同类型的异常。



## mask-image pair



- 参考无监督分割方法，没有真实标签如何对模型分割结果进行评估

- 参考SAM Mask Decoder的iou_score，网络打分
- VLM:使用GPT4V等视觉语言模型进行评估
- mask-image pair 评估指标？





## DM

[视觉扩散模型：调查| 视觉扩散模型IEEE 期刊和杂志 | IEEE探索 --- Diffusion Models in Vision: A Survey | IEEE Journals & Magazine | IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10081412)

Survey

[扩散模型在图像合成方面击败了 GAN --- Diffusion Models Beat GANs on Image Synthesis (neurips.cc)](https://proceedings.neurips.cc/paper_files/paper/2021/hash/49ad23d1ec9fa4bd8d77d02681df5cfa-Abstract.html)

我们表明，扩散模型可以实现优于当前最先进的生成模型的图像样本质量。我们通过一系列消融寻找更好的架构，在无条件图像合成上实现了这一目标。对于条件图像合成，我们通过分类器指导进一步提高样本质量：一种简单、计算高效的方法，使用分类器的梯度来权衡多样性和保真度。我们在 ImageNet 128 × 128 上实现了 2.97，在 ImageNet 256 × 256 上实现了 4.59，在 ImageNet 512 上实现了 7.72 × 512，即使每个样本的前向传递次数只有 25 次，我们仍能匹配 BigGAN 深度，同时保持更好的分布覆盖范围。最后，我们发现分类器指导与上采样扩散模型结合得很好，进一步将 ImageNet 256 × 256 上的 FID 提高到 3.94，在 ImageNet 512 上将 FID 提高到 3.85 

[扩散模型：条件生成classifier guidance，classifier-free guidance (qq.com)](https://mp.weixin.qq.com/s/S46QsAZi9J2LQ_qfx2pu3A)

![image-20240115091813528](https://raw.gitmirror.com/da5sdasddasa/image/main/202401150918577.png)

[[2211.16152\] 小波扩散模型是快速且可扩展的图像生成器 --- [2211.16152] Wavelet Diffusion Models are fast and scalable Image Generators (arxiv.org)](https://arxiv.org/abs/2211.16152)

扩散模型正在成为高保真图像生成的强大解决方案，在许多情况下其质量超过了 GAN。然而，它们缓慢的训练和推理速度是一个巨大的瓶颈，阻碍了它们在实时应用中的使用。最近的 DiffusionGAN 方法通过将采样步骤数从数千减少到几个，显着减少了模型的运行时间，但其速度仍然大大落后于 GAN 同行。本文旨在通过提出一种新颖的基于小波的扩散方案来缩小速度差距。我们通过小波分解从图像和特征级别提取低频和高频分量，并自适应地处理这些分量以实现更快的处理，同时保持良好的生成质量。此外，我们建议使用重建项，这有效地提高了模型训练的收敛性。 CelebA-HQ、CIFAR-10、LSUN-Church 和 STL-10 数据集上的实验结果证明我们的解决方案是提供实时高保真扩散模型的基石。

![image-20240115091128545](https://raw.gitmirror.com/da5sdasddasa/image/main/202401150911267.png)

[[2312.03606\] DiffusionSat：卫星图像的生成基础模型 --- [2312.03606] DiffusionSat: A Generative Foundation Model for Satellite Imagery (arxiv.org)](https://arxiv.org/abs/2312.03606)

扩散模型在图像、语音和视频等多种模式上取得了最先进的结果。然而，现有模型并未专门支持遥感数据，而遥感数据广泛应用于环境监测和作物产量预测等重要应用。卫星图像与自然图像显着不同——它们可以是多光谱的、随时间不规则采样的——而现有的基于网络图像训练的扩散模型不支持它们。此外，遥感数据本质上是时空数据，需要基于字幕或图像的传统方法不支持的条件生成任务。在本文中，我们提出了 DiffusionSat，这是迄今为止在公开的大型高分辨率遥感数据集上训练的最大的生成基础模型。由于卫星图像很少有基于文本的字幕，因此我们将地理位置等相关元数据合并为条件信息。我们的方法产生真实的样本，可用于解决多种生成任务，包括时间生成、给定多光谱输入的超分辨率和修复。我们的方法优于以前最先进的卫星图像生成方法，是第一个用于卫星图像的大规模 generative 基础模型。

![image-20240115091208835](https://raw.gitmirror.com/da5sdasddasa/image/main/202401150912914.png)



## 可靠性评估

![生成](https://raw.gitmirror.com/da5sdasddasa/image/main/202401160949204.svg)





[[2304.01816\]对文本到图像生成的可验证和可重复的人类评估 --- [2304.01816] Toward Verifiable and Reproducible Human Evaluation for Text-to-Image Generation (arxiv.org)](https://arxiv.org/abs/2304.01816) CVPR 2023

[CVPR2023 | 面向文本生成图像的可验证和可复制的人工评估 (qq.com)](https://mp.weixin.qq.com/s/jC7ZwA6g_ue7v5EgXEzaRA)



### **引入人类评价的偏好对齐** RLHF

受ChatGPT通过RLHF的启发，Google Research和加州大学伯克利分校的工作[43]**通过学习人类反馈的激励函数来对齐人类偏好**，实现了更好的颜色、数量、背景生成效果。在Midjourney中，也存在类似的反馈机制，算法往往会生成3张不同的结果，而用户的选择操作实际上就帮助Midjourney完成了对用户偏好的学习。

[[2302.12192\] 使用人工反馈对齐文本到图像模型 --- [2302.12192] Aligning Text-to-Image Models using Human Feedback (arxiv.org)](https://arxiv.org/abs/2302.12192)

用人工反馈对齐文本-图像模型

要点:

1. 提出一种简单有效的方法，通过人工反馈来微调文本到图像模型；
2. 有人工反馈的微调明显改善了图像-文本的对齐，改善幅度高达47%，但代价是图像保真度的轻微下降；
3. 学到的奖励函数比 CLIP 分数更准确地预测了人工的质量评估；
4. 对设计选择的仔细调查对于平衡对齐-保真度的权衡是很重要的。

一句话总结:
用人工反馈微调文本到图像模型，有助于提高图像-文本的一致性。

![图片](https://raw.gitmirror.com/da5sdasddasa/image/main/202401151610291.png)

生成模型在文本到图像的合成方面显示出了令人印象深刻的结果。然而，当前的文本到图像模型通常生成与文本提示不充分对齐的图像。我们提出了一种使用人类反馈来调整此类模型的微调方法，包括三个阶段。首先，我们从一组不同的文本提示中收集人类反馈来评估模型输出对齐情况。然后，我们使用人类标记的图像文本数据集来训练预测人类反馈的奖励函数。最后，通过最大化奖励加权可能性来微调文本到图像模型，以改善图像文本对齐。我们的方法比预先训练的模型更准确地生成具有指定颜色、计数和背景的对象。我们还分析了几种设计选择，并发现对此类设计选择的仔细调查对于平衡对齐保真度权衡非常重要。我们的结果证明了从人类反馈中学习可以显着改进文本到图像模型的潜力。



### VLM代替人工评估

[[2311.01361\] GPT-4V(ision) 作为视觉语言任务的通才评估器 --- [2311.01361] GPT-4V(ision) as a Generalist Evaluator for Vision-Language Tasks (arxiv.org)](https://arxiv.org/abs/2311.01361)

近年来，视觉语言研究取得了重大进展，包括图像到文本生成（Li et al., 2023）、文本到图像生成（Rombach et al., 2021；Ramesh et al., 2021）等方面的进展。 ., 2022），以及图像到图像的翻译（Brooks 等人，2023；Zhang 等人，2023a）。与此同时，人们越来越认识到自动度量在视觉语言任务中的重要性，最近的工作证明了这一点（Hessel et al., 2021b; Hu et al., 2023; Lu et al., 2023; Xu et al., 2023；Wu 等人，2023；Kirstain 等人，2023）。然而，开发符合人类偏好并提供解释性见解的自动指标可能具有挑战性。大多数现有的文本-图像对齐指标专门提供图像和文本特征之间的相似性分数，从而导致三个关键限制：1）无参考约束：现有的专门评分模型大多需要参考（例如，CIDEr（Vedantam et al., 2015） ），这并不总是适用于广泛的任务（例如，文本到图像的合成）； 2）缺乏自由形式的评估：他们不遵守专门的评估指南，无法提供满足多个标准的定量分数或详细的定性解释，这使得其评估难以与人类评估者保持一致，甚至与自身保持一致。 3）单对约束：它们一次只能评估一对图像文本，使其不太适合涉及多个图像文本对的更复杂的任务，例如文本引导的图像编辑。一个问题出现了：是否有一个有前途的通用自动评估器？



• 我们系统地验证了 GPT-4V 作为评估器在各种任务中的能力，包括图像到文本任务（例如图像字幕）、文本到图像任务（例如文本到图像生成） ，图像到图像任务（例如，文本引导的图像编辑）。

• 我们证明，在仔细提示时，GPT-4V 通常是一个很好的无参考、人性化的评估器，这也提供了合理且细粒度的解释。

• 我们发现 GPT-4V 在用作评估器时存在一些局限性，例如感知评估（例如视觉清晰度）和现实世界的复杂案例。



[GPT-4V(ision) is a Human-Aligned Evaluator for Text-to-3D Generation](https://arxiv.org/abs/2401.04092) 

提示工程，使用GPT4V评估两个文本到3D生成模型

尽管文本到 3D 生成方法最近取得了进展，但仍明显缺乏可靠的评估指标。现有指标通常只关注单个标准，例如资产与输入文本的匹配程度。这些指标缺乏对不同评估标准进行概括的灵活性，并且可能与人类偏好不符。进行用户偏好研究是一种替代方法，它既能提供适应性，又能提供与人类偏好一致的结果。然而，用户研究的扩展成本可能非常高。本文提出了一种针对文本到 3D 生成模型的自动、通用且与人类偏好一致的评估指标。为此，我们首先使用 GPT-4V 开发了一个提示生成器来生成评估提示，这些提示用作比较文本到 3D 模型的输入。我们进一步设计了一种方法来指导 GPT-4V 根据用户定义的标准比较两个 3D 资产。最后，我们使用这些成对比较结果为这些模型分配 Elo 等级。实验结果表明，我们的指标在不同的评估标准下与人类偏好高度一致。

![img](https://raw.gitmirror.com/da5sdasddasa/image/main/202401160926306.png)



提示生成管道

我们的目标是设计一个对话管道，该管道可以为 GPT-4V 提供提示生成任务的必要背景，同时保持可定制性，以便评估人员可以根据自己的需要调整此对话。给定一个文本提示和两个由两个不同模型生成的 3D 形状，根据评估标准，判断哪个 3D 形状更好

稳健集成

尽管 GPT-4V 能够对成对形状比较问题提供答案，但由于其推理算法的概率性质，它对相同输入的响应可能会不时发生变化。换句话说，我们可以将 GPT-4V 3D 形状比较器的输出视为分类分布，每个响应都是从分布中抽取的样本。因此，GPT-4V 的单个响应可能无法捕捉到其真正的先验知识，因为它可能会受到采样期间方差的影响。当输出分布的方差较高时尤其如此（例如，当两种选择同样可能时）。请注意，这不是 GPT-4V 的弱点，因为当两个对象根据某个标准同样好时，人类注释员也会发生类似的情况。换句话说，我们对采样 GPT-4V 如何做出决策的一个实例不感兴趣。相反，估计 GPT-4V 选择此答案的概率更有用。

量化性能 ELO得分

我们现在已经获得了一组模型 ℳ 之间的比较列表。这些比较是根据用户定义的标准对各种采样提示 𝒯 进行的。我们的目标现在是使用此信息为 ℳ 中的每个模型分配一个数字，以便最好地解释观察到的结果。如前一小节所述，我们的量化方法应考虑比较结果是从概率分布中抽取的样本这一事实。

这个问题通常在评级国际象棋棋手时进行研究，其中两名棋手之间的比赛即使一方棋手优于另一方棋手，也可能产生不同的结果。在国际象棋和许多其他比赛中，Elo 得分 [19] 可能最广泛采用的方法，可产生反映棋手表现的数值估计。Elo 评级系统也已在先前的工作中进行调整，以评估图像生成模型 [61, 43]。在本文中，我们调整了 Nichol 等人提出的版本 [43]。

