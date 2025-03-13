# Neural Subgraph Counting with Matching Matrix (MMNSC)

A Graph Neural Network (GNN)-based framework for efficient and accurate subgraph counting in large graphs.


## Introduction  
![image](https://github.com/user-attachments/assets/6f593d6a-8d7c-415f-aef4-458c2c31c108)  

MMNSC is a state-of-the-art framework designed for subgraph counting in large graphs. It consists of two key components: **Candidates Extraction** and **Matching Matrix Estimator**.  

1. **Candidates Extraction**:  
   For each vertex in the query graph \( q \), we use the proposed method, **GQL-NS**, to generate a candidate set. Based on the combined candidate sets, we derive candidate substructures from the data graph.  

2. **Matching Matrix Estimator**:  
   Each candidate substructure, along with the query graph and candidate sets, is fed into the matching matrix estimator to obtain a matching matrix for the substructure. Finally, all substructure matrices are combined to produce a matching matrix for the data graph, from which the predicted subgraph count is derived.  


## Installation  

### Dependencies  
The following dependencies are required to run the code:  

- `numpy`  
- `tqdm`  
- `torch>=1.3.0`  
- `scipy`  
- `scikit-learn`  
- `pandas`  
- `networkx`  

You can install the dependencies using the following command:  
```bash
pip install numpy tqdm torch scipy scikit-learn pandas networkx
```

### Clone the Repository  
To get started, clone the repository to your local machine:  
```bash
git clone https://github.com/your-repo/mmnsc.git
cd mmnsc
```



## Usage  

### Running the Model  
To train and evaluate the MMNSC model, use the following command:  
```bash
python code/main.py --graph_file yeast --file_folder ./yeast/data_graph/ --query_path ./yeast/query_graph/ --query_vertex_num 8 --baseline_name _8_baseline.txt --num_epoch 100 --batch_size 20 --learning_rate 0.001
```

#### Command Line Arguments  
- `--graph_file`: The name of the dataset (e.g., `yeast`).  
- `--file_folder`: Path to the directory containing the data graph file.  
- `--query_path`: Path to the directory containing the query graph files.  
- `--query_vertex_num`: The number of vertices in the query graph (e.g., `8`).  
- `--baseline_name`: The suffix of the baseline file containing ground truth subgraph counts (e.g., `_8_baseline.txt`).  
- `--num_epoch`: Number of training epochs (default: `50`).  
- `--batch_size`: Batch size for training (default: `20`).  
- `--learning_rate`: Learning rate for the optimizer (default: `0.01`).  



## Contact  
For any questions or feedback, please contact:  
- Wangrrru: [2232931@mail.dhu.enu.cn](mailto:2232931@mail.dhu.enu.cn)  

