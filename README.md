# shallow-deep-hybrid Journal Paper 
Leveraging Shallow-Deep Hybrid Neural Networks for Optimizing Attack Detection on Edge Servers
Niranjan W. Meegammana, Harinda Fernando
Shilpa Sayura Foundation, Sri Lanka
Sri Lanka Institute of Information Technology, Sri Lanka 
niranjan.meegammana@gmail.com, harinda.f@sliit.lk

Abstract— This study investigates Shallow-Deep Hybrid Fusion Neural Networks (NNs) for detecting network attacks on edge servers. It follows a systematic machine learning workflow to create and evaluate multiple Shallow-Deep Hybrid Fusion models, employing parallel fusion techniques for attack classification. The research combined a single-layer Shallow model with 512 neurons, and a seven-layer Deep model with varying neuron counts to construct 12 Shallow-Deep Hybrid Fusion models by employing weighted sum, concatenation, minimum, maximum, multiplication, and subtraction fusion functions. The hybrid models were trained, validated, and tested using 20 and 40 feature datasets derived from the UNSW-NB15 dataset, and evaluated on performance metrics along with model size, prediction time, memory, and CPU usage.

The results suggest that 20-feature maximum and minimum hybrid fusion models achieved 95.00% accuracy and a precision score of 0.97, applicable for resource-constrained edge servers. The 40-feature concat and maximum hybrid fusion models achieved 98.00% accuracy and a precision score of 0.99, outperforming other models, indicating their suitability for high-resource edge servers. The study proposes exploring the 20-feature hybrid fusion maximum model for federated learning in low-resource edge servers.

In conclusion, Shallow-Deep Hybrid Fusion models are a preferable choice for detecting network attacks on edge servers, and suitable for both high and low-end edge servers. This research contributes valuable insights into enhancing the security of edge servers. Future research will focus on employing these hybrid models with federated learning for securing autonomous vehicle environments. Ultimately, the study suggests the application of Shallow-Deep Hybrid Fusion models beyond cyber security across diverse domains and applications.

Keywords— Shallow-Deep, Parallel Fusion, Hybrid Neural Networks, Attack Detection, Edge Servers
