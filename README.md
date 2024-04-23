## LG-FGAD: An Effective Federated Graph Anomaly Detection Framework
This is the official implementation of LG-FGAD: An Effective Federated Graph Anomaly Detection Framework.

### Dependencies

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

The optional datasets in this code include mix, biochem, molecules and small.

If you've found LG-FGAD useful for your research, please cite our paper as follows:

```
@inproceedings{cai2024lgfgad,
  title={LG-FGAD: An Effective Federated Graph Anomaly Detection Framework},
  author={Jinyu Cai, Yunhe Zhang, Jicong Fan, See-Kiong Ng},
  booktitle={The 33rd International Joint Conference on Artificial Intelligence (IJCAI-24)},
  volume={},
  number={},
  pages={},
  year={2024}
}
```




