# LG-FGAD: An Effective Federated Graph Anomaly Detection Framework

This is an implement of the LG-FGAD paper accepted by the 33rd International Joint Conference on Artificial Intelligence (IJCAI-24).

## Dependencies

- python 3.8, pytorch, torch-geometric, torch-sparse, numpy, scikit-learn, pandas

If you have installed above mentioned packages you can skip this step. Otherwise run:

    pip install -r requirements.txt

## Reproduce graph data results in the single-dataset setting

To generate results

    python LG-FGAD_oneDS.py --data_group DD --eval True

To train LG-FGAD without loading saved weight files

    python LG-FGAD_oneDS.py --data_group DD --eval False

## Reproduce graph data results in the multi-dataset setting

To generate results

    python LG-FGAD_multiDS.py --data_group molecules --eval True

To train LG-FGAD without loading saved weight files

    python LG-FGAD_multiDS.py --data_group molecules --eval False

The optional multi-datasets in this code include mix, biochem, molecules and small.

## Citation

If you use the code or find this repository useful for your research, please consider citing our paper.

```
@inproceedings{cai2024lgfgad,
  title={LG-FGAD: An Effective Federated Graph Anomaly Detection Framework},
  author={Jinyu Cai, Yunhe Zhang, Jicong Fan, See-Kiong Ng},
  booktitle={Proceedings of the 33rd International Joint Conference on Artificial Intelligence (IJCAI-24)},
  pages={3760--3769},
  year={2024}
}
```




