# REFOCUS_Public
Code and data for the REFOCUS paper accepted in SBP'19

Please use this citation when using this data and code:
```
@inproceedings{nazer2019bot,
  title={Bot Detection: Will Focusing on Recall Cause Overall Performance Deterioration?},
  author={Nazer, Tahora H and Davis, Matthew and Karami, Mansooreh and Akoglu, Leman and Koelle, David and Liu, Huan}
  booktitle={International Conference on Social Computing, Behavioral-Cultural Modeling & Prediction and Behavior Representation in Modeling and Simulation (SBP-BRiMS)},
  year={2019}
}
```
Raw data is from 3 publications listed bellow and available as a zip file in this repo:
```
Morstatter, Fred, Liang Wu, Tahora H. Nazer, Kathleen M. Carley, and Huan Liu. "A new approach to bot detection: striking the balance between precision and recall." In 2016 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM), pp. 533-540. IEEE, 2016.
```
```
Lee, Kyumin, Brian David Eoff, and James Caverlee. "Seven months with the devils: A long-term study of content polluters on twitter." In Fifth International AAAI Conference on Weblogs and Social Media. 2011.
```
```
Cresci, Stefano, Roberto Di Pietro, Marinella Petrocchi, Angelo Spognardi, and Maurizio Tesconi. "The paradigm-shift of social spambots: Evidence, theories, and tools for the arms race." In Proceedings of the 26th International Conference on World Wide Web Companion, pp. 963-972. International World Wide Web Conferences Steering Committee, 2017.
```

As the input to the python script, an LDA probability distribution over 200 topics is generated for each user based on the concatination of her/his tweets. You can find the LDA probability distribution files based on each dataset. A sample code for generating such probabilities is included as well.
