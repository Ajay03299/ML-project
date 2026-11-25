# Data attribution and changes

The input panel and sensor CSV were supplied by the team in upstream commit
358297a (data commit 83d15a1). Their bytes are preserved in this contribution.
The panel SHA-256 is 2e438da6caa45da811e8e5c08dd509db5e1c5034eb9038f7381f0755b368daab.

Derived dataset: Ivan Gomez and Sergio Ilarri, Enriched Traffic Datasets for
Madrid, Mendeley Data, Version 4, DOI 10.17632/697ht4f65b.4, CC BY 4.0.
https://data.mendeley.com/datasets/697ht4f65b/4
https://creativecommons.org/licenses/by/4.0/

Original traffic observations and sensor locations: Madrid City Council Open
Data Portal. Historical traffic source:
https://datos.madrid.es/dataset/208627-0-transporte-ptomedida-historico

Road attributes were linked by the dataset authors from OpenStreetMap.
OpenStreetMap contributors: https://www.openstreetmap.org/copyright

Related research article: Gomez and Ilarri, Advanced Prediction of Traffic at
Different Temporal Scales Using Heterogeneous Data Sources, IEEE Open Journal
of Intelligent Transportation Systems (2025), DOI 10.1109/OJITS.2025.3637305.

Experiment changes: select 60 of the 300 supplied sensors; retain the team's
archival missing/long-zero quality screen; construct forecasting features from
past observations; 
apply a chronological split and row budget. Raw aggregation, timezone details and
the historical vintage of static road metadata were not independently verified.
No claim of endorsement by the data authors or city is made.
