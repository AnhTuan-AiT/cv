# Bag-of-Words
Python Implementation of Bag of Words for Image Recognition using OpenCV and
sklearn | [Video](https://www.youtube.com/watch?v=Ba_4wOpbJJM)

Note: see <b>package_version.png</b> in root folder

## Training the classifier
```
py findFeatures.py -t dataset/train/
```

## Testing the classifier
* Testing a number of images
```
py getClass.py -t dataset/test --visualize
```
The `--visualize` flag will display the image with the corresponding label printed on the image/

* Testing a single image
```
py getClass.py -i dataset/test/aeroplane/test_1.jpg --visualize
```

# Troubleshooting

If you get 

```python
AttributeError: 'LinearSVC' object has no attribute 'classes_'
```

then simply retrain the model. 
