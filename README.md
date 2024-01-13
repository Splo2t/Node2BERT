# Node2BERT
## Abstarct
Graph Neural Networks that allow learning on graphs with the structure and the feature information present state-of-the-art by solving diverse real-world problems. However, graph data in the real world often has poor attribute information, furthermore it contains distorted information. It is believed that topological information composed of the relationship between nodes and links in the graph includes the range expressed by the attribute. Therefore, it is necessary to construct a model that is robust to incomplete and absent attributes as well as distorted attribute information by learning the graph only from the structure. We propose Node2BERT, which performs representation learning only with the structure of graphs. Our model consists of a novel search bias model and a BERT model, which learns the graph structure as well as neighbor information. Node2BERT outperforms the existing baseline models in a node classification. In addition, our model displays significant performance when compared to a model using partial attribute information, and the performance gap widens as the attribute-missing ratio deepens.
## Experiments
### Datasets
+ citeseer
+ cora
+ pubmed

| Dataset | Nodes | Edges | Features | Classes | 
| :---: | :---:| :---: | :---: | :---:|
| Citeseer | 3,327 | 4,732 | 3,703 | 6 | 
| Cora | 2,708 | 5,429 | 1,433 | 7 | 
| Pubmed	| 19,717 | 44,324 | 500 | 3 |


A is the adjacency matrix(adding **self-loops**), D is the degree matrix, X is the features, W is the parameters.

### Train
```
python pretraining_eval_ver.py --neighbor_epoch 5 --l "20" --bert_layer "6" --hidden_size 64 --block_size 64 --mlm_prob 0.5 --input cora --position True --seed 44
```

### Node Classification
```
python node_classification_task_simple.py --neighbor_epoch 5 --steps 0 --l "20" --bert_layer "6" --hidden_size 64 --block_size 64 --mlm_prob 0.5 --input cora --position True --seed 44
```

### Requirements
pip install -r requirements.txt

### Experiment results
+ F1-Score using only nodes and edges (Attribute Missing)

| Model | Cora | Citeseer | Pubmed |
|-------|------|----------|--------|
| DeepWalk (Rozemberczki et al., 2021) | 0.833 | 0.603 | 0.802 |
| LINE (Tang et al., 2015) | 0.777 | 0.542 | 0.799 |
| Node2Vec (Grover & Leskovec, 2016) | 0.840 | 0.622 | 0.810 |
| Walklets (Perozzi et al., 2017) | 0.843 | 0.630 | 0.815 |
| NetMF (Qiu et al., 2018) | 0.748 | 0.630 | 0.773 |
| HOPE (Ou et al., 2016) | 0.716 | 0.616 | 0.705 |
| GraRep (Cao et al., 2015) | 0.732 | 0.637 | 0.784 |
| HARP (H. Chen et al., 2018) | 0.786 | 0.548 | 0.698 |
| SAT(GCN) (X. Chen et al., 2020) | 0.788 | 0.661 | 0.405 |
| SAT(GAT) (X. Chen et al., 2020) | 0.817 | 0.667 | 0.450 |
| Node2BERT (ours) | 0.867 | 0.718 | 0.823 |

    

## Reference
```
Rozemberczki, B., Allen, C., & Sarkar, R. (2021). Multi-scale attributed node embedding. Journal of Complex Networks, 9(2), cnab014.
Tang, J., Qu, M., Wang, M., Zhang, M., Yan, J., & Mei, Q. (2015). Line: Large-scale information network embedding. Proceedings of the 24th International Conference on World Wide Web, 1067–1077.
Grover, A., & Leskovec, J. (2016). node2vec: Scalable feature learning for networks. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 855–864.
Perozzi, B., Kulkarni, V., Chen, H., & Skiena, S. (2017). Don’t walk, skip! online learning of multi-scale network embeddings. Proceedings of the 2017 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining 2017, 258–265.
Qiu, J., Dong, Y., Ma, H., Li, J., Wang, K., & Tang, J. (2018). Network embedding as matrix factorization: Unifying deepwalk, line, pte, and node2vec. Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining, 459–467.
Cao, S., Lu, W., & Xu, Q. (2015). Grarep: Learning graph representations with global structural information. Proceedings of the 24th ACM International on Conference on Information and Knowledge Management, 891–900.
Chen, H., Perozzi, B., Hu, Y., & Skiena, S. (2018). Harp: Hierarchical representation learning for networks. Proceedings of the AAAI Conference on Artificial Intelligence, 32(1).
Chen, X., Chen, S., Yao, J., Zheng, H., Zhang, Y., & Tsang, I. W. (2020). Learning on attribute-missing graphs. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(2), 740–757.
https://github.com/ki-ljl/node2vec
```