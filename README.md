# KCQRL: Automatic Skill Extraction and Question Representation Learning for Knowledge Tracing

![The overview of our framework](Framework.png)

This is the repository of KCQRL: Automated **K**nowledge **Concept** Annotation and **Q**uestion **R**epresentation **L**earning for Knowledge Tracing. 

Our KCQRL framework consistently improves the performance of state-of-the-art KT models by a clear margin. For this, we developed our framework in 3 modules: 

1) [KC Annotation](#1-kc-annotation-via-llms): We develop a novel, automated KC annotation approach using large language models (LLMs) that both generates solutions to the questions and labels KCs for each solution step. Thereby, we effectively circumvent the need for manual annotation from domain experts.
2) [Representation Learning of Questions](#2-representation-learning-of-questions): We propose a novel contrastive learning paradigm to jointly learn representations of question content, solution steps, and KCs. As a result, our KCQRL effectively leverages the semantics of question content and KCs, as a clear improvement over existing KT models.
3) [Improving KT Models](#3-improving-kt-models): We integrate the learned representations into KT models to improve their performance. Our framework is flexible and can be combined with any state-of-the-art KT model for improved results.

<table>
  <caption><strong>Improvement in the performance of KT models from our framework.</strong> Shown: AUC with std. dev. across 5 folds. Improvements are shown as both absolute and relative (%) values.</caption>
  <thead>
    <tr>
      <th>Model</th>
      <th colspan="4">XES3G5M</th>
      <th colspan="4">Eedi</th>
    </tr>
    <tr>
      <th></th>
      <th>Default</th>
      <th>w/ framework</th>
      <th>Imp. (abs.)</th>
      <th>Imp. (%)</th>
      <th>Default</th>
      <th>w/ framework</th>
      <th>Imp. (abs.)</th>
      <th>Imp. (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>DKT</td>
      <td>78.33 ± 0.06</td>
      <td>82.13 ± 0.02</td>
      <td style="background-color: #228B22;">+3.80</td>
      <td style="background-color: #228B22;">+4.85%</td>
      <td>73.59 ± 0.01</td>
      <td>74.97 ± 0.03</td>
      <td style="background-color: #66CD66;">+1.38</td>
      <td style="background-color: #66CD66;">+1.88%</td>
    </tr>
    <tr>
      <td>DKT$+$</td>
      <td>78.57 ± 0.05</td>
      <td>82.34 ± 0.04</td>
      <td style="background-color: #228B22;">+3.77</td>
      <td style="background-color: #228B22;">+4.80%</td>
      <td>73.79 ± 0.03</td>
      <td>75.32 ± 0.04</td>
      <td style="background-color: #228B22;">+1.53</td>
      <td style="background-color: #228B22;">+2.07%</td>
    </tr>
    <tr>
      <td>KQN</td>
      <td>77.81 ± 0.03</td>
      <td>82.10 ± 0.06</td>
      <td style="background-color: #228B22;">+4.29</td>
      <td style="background-color: #228B22;">+5.51%</td>
      <td>73.13 ± 0.01</td>
      <td>75.16 ± 0.04</td>
      <td style="background-color: #228B22;">+2.03</td>
      <td style="background-color: #228B22;">+2.78%</td>
    </tr>
    <tr>
      <td>qDKT</td>
      <td>81.94 ± 0.05</td>
      <td>82.13 ± 0.05</td>
      <td style="background-color: #66CD66;">+0.19</td>
      <td style="background-color: #66CD66;">+0.23%</td>
      <td>74.09 ± 0.03</td>
      <td>74.97 ± 0.04</td>
      <td style="background-color: #66CD66;">+0.88</td>
      <td style="background-color: #66CD66;">+1.19%</td>
    </tr>
    <tr>
      <td>IEKT</td>
      <td><strong>82.24 ± 0.07</strong></td>
      <td>82.82 ± 0.06</td>
      <td style="background-color: #66CD66;">+0.58</td>
      <td style="background-color: #66CD66;">+0.71%</td>
      <td>75.12 ± 0.02</td>
      <td>75.56 ± 0.02</td>
      <td style="background-color: #66CD66;">+0.44</td>
      <td style="background-color: #66CD66;">+0.59%</td>
    </tr>
    <tr>
      <td>AT-DKT</td>
      <td>78.36 ± 0.06</td>
      <td>82.36 ± 0.07</td>
      <td style="background-color: #228B22;">+4.00</td>
      <td style="background-color: #228B22;">+5.10%</td>
      <td>73.72 ± 0.04</td>
      <td>75.25 ± 0.02</td>
      <td style="background-color: #228B22;">+1.53</td>
      <td style="background-color: #228B22;">+2.08%</td>
    </tr>
    <tr>
      <td>QIKT</td>
      <td>82.07 ± 0.04</td>
      <td>82.62 ± 0.05</td>
      <td style="background-color: #66CD66;">+0.55</td>
      <td style="background-color: #66CD66;">+0.67%</td>
      <td><strong>75.15 ± 0.04</strong></td>
      <td>75.74 ± 0.02</td>
      <td style="background-color: #66CD66;">+0.59</td>
      <td style="background-color: #66CD66;">+0.79%</td>
    </tr>
    <tr>
      <td>DKVMN</td>
      <td>77.88 ± 0.04</td>
      <td>82.64 ± 0.02</td>
      <td style="background-color: #228B22;">+4.76</td>
      <td style="background-color: #228B22;">+6.11%</td>
      <td>72.74 ± 0.05</td>
      <td>75.51 ± 0.02</td>
      <td style="background-color: #228B22;">+2.77</td>
      <td style="background-color: #228B22;">+3.81%</td>
    </tr>
    <tr>
      <td>DeepIRT</td>
      <td>77.81 ± 0.06</td>
      <td>82.56 ± 0.02</td>
      <td style="background-color: #228B22;">+4.75</td>
      <td style="background-color: #228B22;">+6.10%</td>
      <td>72.61 ± 0.02</td>
      <td>75.18 ± 0.05</td>
      <td style="background-color: #228B22;">+2.57</td>
      <td style="background-color: #228B22;">+3.54%</td>
    </tr>
    <tr>
      <td>ATKT</td>
      <td>79.78 ± 0.07</td>
      <td>82.37 ± 0.04</td>
      <td style="background-color: #228B22;">+2.59</td>
      <td style="background-color: #228B22;">+3.25%</td>
      <td>72.17 ± 0.03</td>
      <td>75.28 ± 0.04</td>
      <td style="background-color: #228B22;">+3.11</td>
      <td style="background-color: #228B22;">+4.31%</td>
    </tr>
    <tr>
      <td>SAKT</td>
      <td>75.90 ± 0.05</td>
      <td>81.64 ± 0.03</td>
      <td style="background-color: #228B22;">+5.74</td>
      <td style="background-color: #228B22;">+7.56%</td>
      <td>71.60 ± 0.03</td>
      <td>74.77 ± 0.02</td>
      <td style="background-color: #228B22;">+3.17</td>
      <td style="background-color: #228B22;">+4.43%</td>
    </tr>
    <tr>
      <td>SAINT</td>
      <td>79.65 ± 0.02</td>
      <td>81.50 ± 0.07</td>
      <td style="background-color: #228B22;">+1.85</td>
      <td style="background-color: #228B22;">+2.32%</td>
      <td>73.96 ± 0.02</td>
      <td>75.20 ± 0.04</td>
      <td style="background-color: #66CD66;">+1.24</td>
      <td style="background-color: #66CD66;">+1.68%</td>
    </tr>
    <tr>
      <td>AKT</td>
      <td>81.67 ± 0.03</td>
      <td><strong>83.04 ± 0.05</strong></td>
      <td style="background-color: #66CD66;">+1.37</td>
      <td style="background-color: #66CD66;">+1.68%</td>
      <td>74.27 ± 0.03</td>
      <td>75.49 ± 0.03</td>
      <td style="background-color: #66CD66;">+1.22</td>
      <td style="background-color: #66CD66;">+1.64%</td>
    </tr>
    <tr>
      <td>simpleKT</td>
      <td>81.05 ± 0.06</td>
      <td>82.92 ± 0.04</td>
      <td style="background-color: #228B22;">+1.87</td>
      <td style="background-color: #228B22;">+2.31%</td>
      <td>73.90 ± 0.04</td>
      <td>75.46 ± 0.02</td>
      <td style="background-color: #228B22;">+1.56</td>
      <td style="background-color: #228B22;">+2.11%</td>
    </tr>
    <tr>
      <td>sparseKT</td>
      <td>79.65 ± 0.11</td>
      <td>82.95 ± 0.09</td>
      <td style="background-color: #228B22;">+3.30</td>
      <td style="background-color: #228B22;">+4.14%</td>
      <td>74.98 ± 0.09</td>
      <td><strong>78.96 ± 0.08</strong></td>
      <td style="background-color: #228B22;">+3.98</td>
      <td style="background-color: #228B22;">+5.31%</td>
    </tr>
  </tbody>
  <tfoot>
    <tr>
      <td colspan="9">Best values are in bold. The shading in green shows the magnitude of the performance gain.</td>
    </tr>
  </tfoot>
</table>

## Setup
 
 **Dataset details**: We used XES3G5M (we translated from Chinese to English) and EEDI datasets for our work. 

 - The details of XES3G5M can be found [here](https://github.com/ai4ed/XES3G5M). You can download the dataset by following instructions there. After the download, You can add the files from [data/XES3G5M/metadata](data/XES3G5M/metadata) to run our framework. 
 - [EEDI](https://eedi.com) dataset can be acquired upon request 

 `Important note`: For XES3G5M, we already provide its English translation, entire output from our KC annotation, and the clustering of KCs [here](data/XES3G5M/metadata).
 * Therefore, after downloading XES3G5M dataset from its source (for exercise histories), you can directly start from our [Representation Learning of Questions](#2-representation-learning-of-questions) and quickly improve your existing KT model!

 **Python environment**: We used Python 3.11.6 in our implementation. We use two separate virtual environments in our framework. 

 - Install the libraries via `pip install -r requirements_env_rl.txt` for [KC Annotation](#1-kc-annotation-via-llms) and [Representation Learning](#2-representation-learning-of-questions)

 - Install the libraries via `pip install -r requirements_env_pykt.txt` for [Improving KT models](#3-improving-kt-models). After loading libraries, locate [pykt-toolkit](pykt-toolkit) and run the command `pip install -e .` to install our custom version of pykt with improved kt implementations. 

 ## 1) KC Annotation via LLMs

 This part shows an example usage of full KC annotation pipeline. To run the scripts, first locate [kc_annotation](kc_annotation) folder
 
We use the English translation of XES3G5M dataset [`questions_translated.json`](data/XES3G5M/metadata/questions_translated.json) as our running example. 

### a) Solution step generation 

You can run the command below

`python get_mapping_kc_solsteps.py --original_question_file ../data/XES3G5M/metadata/questions_translated.json --annotated_question_file ../data/XES3G5M/metadata/questions_translated_kc_annotated.json`

### b) KC annotation

You can run the command below

`python get_kc_annotation.py --original_question_file ../data/XES3G5M/metadata/questions_translated_kc_annotated.json --annotated_question_file ../data/XES3G5M/metadata/questions_translated_kc_sol_annotated.json`

### c) Solution Step - KC mapping

You can run the command below

`python get_mapping_kc_solsteps.py --original_question_file ../data/XES3G5M/metadata/questions_translated_kc_sol_annotated.json --mapped_question_file ../data/XES3G5M/metadata/questions_translated_kc_sol_annotated_mapped.json`

**Note:** For convenience, we provide the final output of this pipeline  [questions_translated_kc_sol_annotated_mapped.json](../data/XES3G5M/metadata/questions_translated_kc_sol_annotated_mapped.json).

## 2) Representation Learning of Questions

For this part, please locate [representation_learning](representation_learning) folder.

For training, you can run the command below:

`python train.py --json_file_dataset ../data/XES3G5M/metadata/questions_translated_kc_sol_annotated_mapped.json --json_file_cluster_kc data/XES3G5M/metadata/kc_clusters_hdbscan.json --json_file_kc_questions data/XES3G5M/metadata/kc_questions_map.json --wandb_project_name <your_wandb_project_name>`

Note that the above command requires you to setup your wandb account first. 

After training, you can save the embeddings by following [save_embeddings.ipynb](representation_learning/save_embeddings.ipynb). 

## 3) Improving KT Models

We implemented the improved versions of KT models via [pykt](https://github.com/pykt-team/pykt-toolkit) library. We forked the library to [pykt-toolkit](pykt-toolkit) and developed the algorithsm there. Specifically, our implemented KT models can be found in [models](pykt-toolkit/pykt/models) folder. 

As the naming convention, we added `Que` suffix to the existing models, where "que" refers to our "learned question representations". For instance, the improved version of `SimpleKT` is implemented as `SimpleKTQue` and can be found in [simplekt_que.py](pykt-toolkit/pykt/models/simplekt_que.py). 

For training the these models, you can locate [train_test](pykt-toolkit/train_test) folder. You can train `SimpleKTQue` with the command below: 

`python sparsekt_que_train.py --emb_path <embeddings_from_representation_learning>`

Note that the above command requires you to setup your wandb account first. 

We use `wandb_eval.py` and `wandb_predict.py` from pykt library for evaluation. The details of the library can be found in their [documentation](https://pykt-toolkit.readthedocs.io/en/latest/).