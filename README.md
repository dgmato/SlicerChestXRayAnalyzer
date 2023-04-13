# SlicerChestXRayAnalyzer
Tools to analyze chest X ray images using AI tools

## Prerequisites
To install the requirements file in the Python console inside Slicer, you can follow these steps:</br>
```python
import os
path='absolute/path/to/directory/containing/requirements.txt'
os.chdir(path)
slicer.util.pip_install('-r requirements.txt')
```