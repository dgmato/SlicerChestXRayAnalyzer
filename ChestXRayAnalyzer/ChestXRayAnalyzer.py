import csv
import logging
import math
import numpy as np
import os
import random
import qt
import time
import vtk

import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

# torchXRayVision library
import torchxrayvision as xrv

import torch
import torchvision
import cv2
from PIL import Image
import numpy as np
import skimage

from preprocess.preprocess import preprocess
        
        
#------------------------------------------------------------------------------
#
# ChestXRayAnalyzer
#
#------------------------------------------------------------------------------
class ChestXRayAnalyzer(ScriptedLoadableModule):
    
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Chest X-Ray Analyzer"
        self.parent.categories = ["Chest X-Ray Analyzer"]
        self.parent.dependencies = []
        self.parent.contributors = ["David Garcia Mato (Ebatinca S.L.)"]
        self.parent.helpText = """
        Module to analyze chest X-ray images using AI tools.
        """
        self.parent.acknowledgementText = """
        This file was originally developed by David Garcia Mato (Ebatinca S.L.).
        """


#------------------------------------------------------------------------------
#
# ChestXRayAnalyzerWidget
#
#------------------------------------------------------------------------------
class ChestXRayAnalyzerWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    
    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False

        # TESTING
        slicer.ChestXRayAnalyzerWidget = self

    #-------------------------------------------------------------------------------
    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/ChestXRayAnalyzer.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = ChestXRayAnalyzerLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).
        self.ui.inputVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.segmentationModelComboBox.currentTextChanged.connect(self.updateParameterNodeFromGUI)
        self.ui.classificationModelComboBox.currentTextChanged.connect(self.updateParameterNodeFromGUI)

        # Buttons
        self.ui.runButton.connect('clicked(bool)', self.onRunButton)
        self.ui.classifyButton.connect('clicked(bool)', self.onClassifyButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

        # Set ProgressBar to 0
        self.setProgressBarValues()

    #-------------------------------------------------------------------------------
    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    #-------------------------------------------------------------------------------
    def enter(self):
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    #-------------------------------------------------------------------------------
    def exit(self):
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    #-------------------------------------------------------------------------------
    def onSceneStartClose(self, caller, event):
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    #-------------------------------------------------------------------------------
    def onSceneEndClose(self, caller, event):
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    #-------------------------------------------------------------------------------
    def initializeParameterNode(self):
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.GetNodeReference("InputVolume"):
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())

    #-------------------------------------------------------------------------------
    def setParameterNode(self, inputParameterNode):
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """
        if inputParameterNode:
            self.logic.setDefaultParameters(inputParameterNode)

        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None:
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

        # Initial GUI update
        self.updateGUIFromParameterNode()

    #-------------------------------------------------------------------------------
    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """
        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True

        # Update node selectors
        self.ui.inputVolumeSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume"))

        # Update model selector
        self.ui.segmentationModelComboBox.setCurrentText(self._parameterNode.GetParameter("SelectedSegmentationModel"))
        self.ui.classificationModelComboBox.setCurrentText(self._parameterNode.GetParameter("SelectedClassificationModel"))

        # Update buttons states and tooltips
        if self._parameterNode.GetNodeReference("InputVolume"):
            self.ui.runButton.toolTip = "Run automatic segmentation."
            self.ui.runButton.enabled = True
            self.ui.classifyButton.toolTip = "Run automatic classification."
            self.ui.classifyButton.enabled = True
        else:
            self.ui.runButton.toolTip = "Select input volume."
            self.ui.runButton.enabled = False
            self.ui.classifyButton.toolTip = "Select input volume."
            self.ui.classifyButton.enabled = False

        # All the GUI updates are done
        self._updatingGUIFromParameterNode = False

    #-------------------------------------------------------------------------------
    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """
        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

        self._parameterNode.SetNodeReferenceID("InputVolume", self.ui.inputVolumeSelector.currentNodeID)
        self._parameterNode.SetParameter("SelectedSegmentationModel", self.ui.segmentationModelComboBox.currentText)
        self._parameterNode.SetParameter("SelectedClassificationModel", self.ui.classificationModelComboBox.currentText)

        self._parameterNode.EndModify(wasModified)

        # Dispaly the selected volume in Red Slice View
        inputVolume = self._parameterNode.GetNodeReference("InputVolume")
        if inputVolume:
            self.logic.displayVolumeInSliceView(inputVolume)

    #-------------------------------------------------------------------------------
    def onRunButton(self):
        """
        Run processing.
        """
        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor = True):
            # Run inference
            self.logic.run(self._parameterNode)

    #-------------------------------------------------------------------------------
    def onClassifyButton(self):
        """
        Run processing.
        """
        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor = True):
            # Run inference
            self.logic.runClassifierValenciaModels(self._parameterNode)
            self.setProgressBarValues()

    #-------------------------------------------------------------------------------
    def setProgressBarValues(self):
        for pathology, path in self.logic.model_paths.items():
            progress_bar_name = pathology + '_progressBar'
            progress_bar = getattr(self.ui, progress_bar_name)
            self.logic.updateProgressBar(progress_bar, pathology)

#------------------------------------------------------------------------------
#
# ChestXRayAnalyzerLogic
#
#------------------------------------------------------------------------------
class ChestXRayAnalyzerLogic(ScriptedLoadableModuleLogic):
    
    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)

        # Red slice
        self.redSliceLogic = slicer.app.layoutManager().sliceWidget("Red").sliceLogic()

        # Output segmentation
        self.outputSegmentation = None

        # Classification results
        self.results = None

        # Segment colors
        self.rightLungColor = (0.5, 0.68, 0.5)
        self.leftLungColor = (0.95, 0.84, 0.57)

        # Load sample data
        loadSampleDataEnabled = False
        if loadSampleDataEnabled:
            volumeFilePath = slicer.modules.ChestXRayAnalyzerWidget.resourcePath('SampleData/100469495785351489872749036114751610212_rfyvv7.png')
            try:
                volumeNode = slicer.util.loadVolume(volumeFilePath)
            except:
                print('ERROR: sample data could not be loaded from: ', volumeFilePath)

        # Image path (used to process the image)
        self.img_path = slicer.modules.ChestXRayAnalyzerWidget.resourcePath('image.png')

        # Model path
        self.model_folder_path = slicer.modules.ChestXRayAnalyzerWidget.resourcePath('Model/Pathology_classification')
        self.model_paths = {
            'alveolar': 'alveolar_pattern-resnet18.pth',
            'atelectasis': 'atelectasis-resnet18.pth',
            'cardiomegaly':'cardiomegaly-resnet18.pth',
            'consolidation':'consolidation-resnet18.pth',
            'covid19':'covid19-resnet18.pth',
            'infiltrates':'infiltrates-resnet18.pth',
            'nodule': 'nodule-resnet18.pth',
            'normal': 'normal-resnet18.pth',
            'pleural_effusion':'pleural_effusion-resnet18.pth',
            'pneumonia':'pneumonia-resnet18.pth',
            'pseudonodule': 'pseudonodule-resnet18.pth',
            'pulmonary_mass': 'pulmonary_mass-resnet18.pth',
        }

        # Set red slice only layout
        slicer.app.layoutManager().setLayout(6)

    #-------------------------------------------------------------------------------
    def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """
        parameterNode.SetParameter("SelectedSegmentationModel", "Pyramid Scene Parsing (PSP) Network")
        parameterNode.SetParameter("SelectedClassificationModel", "resnet50-res512-all")

    #-------------------------------------------------------------------------------
    def run(self, parameterNode):
        """
        Compute automatic segmentation of lungs.

        :param inputVolume: CT chest volume
        """
        if not parameterNode:
            return

        # Get initial timestamp
        startTime = time.time()
        logging.info('Processing started')

        # Crop input volume
        self.centerCropVolume(parameterNode)

        # Automatic segmentation
        self.runAutomaticSegmentation(parameterNode)

        # Classification
        self.runClassifier(parameterNode)

        # Get final timestamp
        stopTime = time.time()

        # Report processing time
        logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')
        print(f'Processing completed in {stopTime-startTime:.2f} seconds')

        return True

    #-------------------------------------------------------------------------------
    def centerCropVolume(self, parameterNode):
        if not parameterNode:
            return

        # Get input volume
        try:
            inputVolumeNode = parameterNode.GetNodeReference("InputVolume")
        except:
            logging.error('Input volume was not defined.')
            return

        # Get array
        volumeArray = slicer.util.arrayFromVolume(inputVolumeNode)

        # Crop center square of input image
        centerCropTransform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop()])
        volumeArray_centerCrop = centerCropTransform(volumeArray)

        # Update volume node with cropped array
        slicer.util.updateVolumeFromArray(inputVolumeNode, volumeArray_centerCrop)
    
    #-------------------------------------------------------------------------------
    def runAutomaticSegmentation(self, parameterNode):
        if not parameterNode:
            return

        # Get input volume
        try:
            inputVolumeNode = parameterNode.GetNodeReference("InputVolume")
        except:
            logging.error('Input volume was not defined.')
            return
            
        # Get segmentation model
        try:
            selectedSegmentationModel = parameterNode.GetParameter("SelectedSegmentationModel")
        except:
            logging.error('Segmentation model was not selected.')
            return
            
        # Create progress dialog and store start time
        progressDialog = slicer.util.createProgressDialog(parent=slicer.util.mainWindow(), value=0, maximum=100)

        # Set wait cursor
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        
        # Clear previous segmentation if any. Otherwise create new one
        if self.outputSegmentation:
            self.outputSegmentation.GetSegmentation().RemoveAllSegments()
        else:
            self.outputSegmentation = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "Segmentation")
            self.outputSegmentation.CreateDefaultDisplayNodes()
        
        # Get array
        volumeArray = slicer.util.arrayFromVolume(inputVolumeNode)

        # Save original image size
        numRows = volumeArray.shape[2]
        numColumns = volumeArray.shape[1]

        # Convert 8-bit image to [-1024, 1024] range
        img = xrv.datasets.normalize(volumeArray, volumeArray.max())

        # Apply segmentation model
        if selectedSegmentationModel == 'Pyramid Scene Parsing (PSP) Network':
            # Resize image to 512 x 512
            transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(512)])
            img = transform(img)
            img = torch.from_numpy(img)
            # Get segmentation model
            model = xrv.baseline_models.chestx_det.PSPNet()
        else:
            logging.error('Invalid segmentation model.')
            return

        # Run inference
        with torch.no_grad():
            pred = model(img)

        # model.targets --> ['Left Clavicle', 'Right Clavicle', 'Left Scapula', 'Right Scapula', 'Left Lung', 'Right Lung', 'Left Hilus Pulmonis', 'Right Hilus Pulmonis', 'Heart', 'Aorta', 'Facies Diaphragmatica', 'Mediastinum', 'Weasand', 'Spine']

        # Convert prediction to mask
        pred = 1 / (1 + np.exp(-pred))  # sigmoid
        pred[pred < 0.5] = 0
        pred[pred > 0.5] = 1

        # Convert prediction to segments
        segmentationProgressStep = 100/(len(model.targets))
        for targetId, targetName in enumerate(model.targets):
            # Report progress
            self.progress(progressDialog, segmentationProgressStep * (targetId+1), f'Running automatic segmentation...')
            # Resize mask
            targetMask = np.array(pred[0,targetId])
            targetMask_resize = cv2.resize(targetMask, (numRows, numColumns))
            targetMask_expand = np.expand_dims(targetMask_resize, 0)
            # Assign color to segment
            if targetName == 'Left Lung':
                segmentColor = self.leftLungColor
            elif targetName == 'Right Lung':
                segmentColor = self.rightLungColor
            else: # random
                segmentColor = [random.uniform(0.0, 1.0), random.uniform(0.0, 1.0), random.uniform(0.0, 1.0)]
            # Add target as segment
            self.addSegmentFromNumpyArray(self.outputSegmentation, targetMask_expand, targetName, 1, inputVolumeNode, segmentColor)

        # Hide all segments except for lungs
        segmentOpacity = 0.9
        numSegments = self.outputSegmentation.GetSegmentation().GetNumberOfSegments()
        for segmentPos in range(numSegments):
            currentSegmentID = self.outputSegmentation.GetSegmentation().GetNthSegmentID(segmentPos)
            self.outputSegmentation.GetDisplayNode().SetSegmentOpacity(currentSegmentID, segmentOpacity)
            if (currentSegmentID == 'Left Lung') or (currentSegmentID == 'Right Lung'):
                self.outputSegmentation.GetDisplayNode().SetSegmentVisibility(currentSegmentID, True)
                #self.outputSegmentation.GetDisplayNode().SetSegmentOpacity(currentSegmentID, segmentOpacity)
                self.outputSegmentation.GetDisplayNode().SetSegmentOpacity2DOutline(currentSegmentID, 1.0)
                self.outputSegmentation.GetDisplayNode().SetSegmentOpacity2DFill(currentSegmentID, 0.1)
            else:
                self.outputSegmentation.GetDisplayNode().SetSegmentVisibility(currentSegmentID, False)

        # End processing
        qt.QApplication.restoreOverrideCursor()
        progressDialog.close()

    #-------------------------------------------------------------------------------
    def runClassifier(self, parameterNode):
        if not parameterNode:
            return

        # Get input volume
        try:
            inputVolumeNode = parameterNode.GetNodeReference("InputVolume")
        except:
            logging.error('Input volume was not defined.')
            return

        # Get classification model
        try:
            selectedClassificationModel = parameterNode.GetParameter("SelectedClassificationModel")
        except:
            logging.error('Classification model was not selected.')
            return        
            
        # Create progress dialog and store start time
        progressDialog = slicer.util.createProgressDialog(parent=slicer.util.mainWindow(), value=0, maximum=100)

        # Set wait cursor
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        
        # Get array
        volumeArray = slicer.util.arrayFromVolume(inputVolumeNode)

        # Save original image size
        numRows = volumeArray.shape[2]
        numColumns = volumeArray.shape[1]

        # Convert 8-bit image to [-1024, 1024] range
        img = xrv.datasets.normalize(volumeArray, volumeArray.max())

        # Apply classification model
        if selectedClassificationModel == 'resnet50-res512-all':
            # Resize image to 512 x 512
            transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(512)])
            img = transform(img)
            img = torch.from_numpy(img)
            # Get classification model
            model = xrv.models.ResNet(weights="resnet50-res512-all")
            print('Classification model = ', model)

        elif selectedClassificationModel == 'densenet121-res224-all':
            # Resize image to 224 x 224
            transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])
            img = transform(img)
            img = torch.from_numpy(img)
            # Get classification model
            model = xrv.models.DenseNet(weights="densenet121-res224-all")
            print('Classification model = ', model)

        else:
            logging.error('Invalid segmentation model.')
            return        

        # model.pathologies
        # ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 
        # 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 
        # 'Mass', 'Hernia', 'Lung Lesion', 'Fracture', 'Lung Opacity', 'Enlarged Cardiomediastinum']

        # Run inference
        outputs = model(img[None,...])

        # Print results
        print(dict(zip(model.pathologies,outputs[0].detach().numpy())))

        # Compute pathology heatmaps
        computeHeatmaps = True
        shNode = slicer.mrmlScene.GetSubjectHierarchyNode() # hierarchy node
        heatmapFolderId = shNode.CreateFolderItem(shNode.GetSceneItemID(), 'Heatmaps') # create folder to store heatmaps

        # Enable gradient computation for this tensor
        inputImage = img[None,...]
        inputImage = inputImage.requires_grad_()

        # Run inference
        outputs = model(inputImage)
        #print(outputs.shape) 
        
        if computeHeatmaps:
            heatmap_dict = {}
            heatmapProgressStep = 100/(len(model.pathologies))
            for pathologyId, pathologyName in enumerate(model.pathologies):
                print('Computing heatmap for pathology: ', pathologyName)
                # Report progress
                self.progress(progressDialog, heatmapProgressStep * (pathologyId+1), f'Running pathology classication...')               
                # Compute heatmap
                heatmap = self.getPathologyHeatmap(inputImage, outputs, model, pathologyName)
                #print('   - Shape: ', heatmap.shape)             
                # Store heatmap
                heatmap_dict[pathologyName] = heatmap

                # Resize heatmap
                heatmap = np.array(heatmap)
                heatmap_resize = cv2.resize(heatmap, (numRows, numColumns))
                heatmap_expand = np.expand_dims(heatmap_resize, 0)

                # Normalize heatmap
                heatmap_norm = xrv.datasets.normalize(heatmap_expand, heatmap_expand.max())
                
                # Create heatmap volume node
                inputVolumeItemID = shNode.GetItemByDataNode(inputVolumeNode)
                heatmapVolumeItemID = slicer.modules.subjecthierarchy.logic().CloneSubjectHierarchyItem(shNode, inputVolumeItemID)
                heatmapVolumeNode = shNode.GetItemDataNode(heatmapVolumeItemID)
                heatmapVolumeNode.SetName('Heatmap - ' + pathologyName)
                slicer.util.updateVolumeFromArray(heatmapVolumeNode, heatmap_norm)

                # Modify display properties
                heatmapVolumeDisplayNode = heatmapVolumeNode.GetDisplayNode()
                heatmapVolumeDisplayNode.SetAndObserveColorNodeID('vtkMRMLPETProceduralColorNodePET-Heat') # lookup table
                heatmapVolumeDisplayNode.SetAutoThreshold(True)
                heatmapVolumeDisplayNode.SetAutoWindowLevel(True)

                # Store heatmap volume node in folder
                shNode.SetItemParent(heatmapVolumeItemID, heatmapFolderId)

        # End processing
        qt.QApplication.restoreOverrideCursor()
        progressDialog.close()
        
    #-------------------------------------------------------------------------------
    def getPathologyHeatmap(self, image, outputs, model, pathologyName):
        target = model.pathologies.index(pathologyName)
        grads = torch.autograd.grad(outputs[:,target], image, retain_graph = True)[0][0][0]
        blurred = skimage.filters.gaussian(grads.detach().cpu().numpy()**2, sigma=(5, 5), truncate=3.5)
        return blurred

    #-------------------------------------------------------------------------------
    def runClassifierValenciaModels(self, parameterNode):
        if not parameterNode:
            return

        startTime = time.time()
        print('Processing started')

        # Get input volume
        try:
            inputVolumeNode = parameterNode.GetNodeReference("InputVolume")
        except:
            logging.error('Input volume was not defined.')
            return

        # Get classification models
        models = {}
        for pathology, model_path in self.model_paths.items():
            models[pathology] = torch.load(os.path.join(self.model_folder_path, model_path), map_location = 'cpu')
            models[pathology].eval()
        
        # Create progress dialog and store start time
        progressDialog = slicer.util.createProgressDialog(parent=slicer.util.mainWindow(), value=0, maximum=100)

        # Set wait cursor
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)

        # Get array
        volumeArray = slicer.util.arrayFromVolume(inputVolumeNode)

        # Preprocess image
        transform = torchvision.transforms.Compose([
                            torchvision.transforms.Resize((1024,1024)),
                            torchvision.transforms.ToTensor()
                            ])

        cv2.imwrite(self.img_path, volumeArray.squeeze())
        img = preprocess(self.img_path, "MONOCHROME2")
        img = img[...,::-1]
        img_trans = transform(Image.fromarray(img)).unsqueeze(0)

        # Run inference
        print("--------------Predicted Results-------------")
        self.results = {}

        for pathology, model in models.items():
            r = {}
            r['out'] = model(img_trans)

            prob = torch.nn.functional.softmax(r['out'], dim=1).data.squeeze()
            r['probs'] = prob.cpu().numpy()

            self.results[pathology] = r 
            print(pathology,':', self.results[pathology]['probs'][1])
        
        # End processing
        qt.QApplication.restoreOverrideCursor()
        progressDialog.close()  

        # Get final timestamp
        stopTime = time.time()

        # Report processing time
        print(f'Processing completed in {stopTime-startTime:.2f} seconds')
        
    #-------------------------------------------------------------------------------
    def updateProgressBar(self, progress_bar,pathology):
        if self.results == None: 
            progress_bar.setValue(0)
        else:
            percentage_predicted = self.results[pathology]['probs'][1] * 100
            progress_bar.setValue(percentage_predicted)

            # Set ProgressBar Color
            if (pathology != 'normal' and percentage_predicted > 50) or (pathology== 'normal' and percentage_predicted < 50):
                progress_bar.setStyleSheet("QProgressBar::chunk {background-color: red;}")
            else:
                progress_bar.setStyleSheet("QProgressBar::chunk {background-color: green;}")
                
    #-------------------------------------------------------------------------------
    def loadModel(self):
        # Weights file path
        weightsFilePath = slicer.modules.ChestXRayAnalyzerWidget.resourcePath('Models') + '/deepedit_dynunet.pt'

        # Define network
        self.labels = {
                    "mandible": 1,
                    "left inferior alveolar nerve": 36,
                    "right inferior alveolar nerve": 37,
                    "background": 0,
                }
        self.number_intensity_ch = 1
        self.network = DynUNet(spatial_dims=3,
                    in_channels=len(self.labels) + self.number_intensity_ch,
                    out_channels=len(self.labels),
                    kernel_size=[3, 3, 3, 3, 3, 3],
                    strides=[1, 2, 2, 2, 2, [2, 2, 1]],
                    upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
                    norm_name="instance",
                    deep_supervision=False,
                    res_block=True,
                )

        # Load weights 
        try:
            self.network.load_state_dict(torch.load(weightsFilePath, map_location = DEVICE)) 
            print('\nModel loaded from file:', weightsFilePath)     
        except:
            logging.error("\nFailed to load model.")
            return
    
    #-------------------------------------------------------------------------------
    def addSegmentFromNumpyArray(self, outputSegmentation, input_np, segmentName, labelValue, inputVolume, color):
        emptySegment = slicer.vtkSegment()
        emptySegment.SetName(segmentName)
        emptySegment.SetColor(color)
        outputSegmentation.GetSegmentation().AddSegment(emptySegment)

        segment_np = np.zeros(input_np.shape)
        segment_np[ input_np==labelValue ] = 1

        segmentId = self.outputSegmentation.GetSegmentation().GetSegmentIdBySegmentName(segmentName)
        slicer.util.updateSegmentBinaryLabelmapFromArray(segment_np, outputSegmentation, segmentId, inputVolume)    

    #------------------------------------------------------------------------------
    def progress(self, progressDialog, progressPercent, message):
        """
        Utility function to change text in the progress dialog, increase progress value, and force render.
        """
        progressDialog.setValue(progressPercent)
        progressDialog.labelText = message
        slicer.app.processEvents()
        if progressDialog.wasCanceled:
            raise CancelledError('Cancelled')

    #------------------------------------------------------------------------------
    def displayVolumeInSliceView(self, volumeNode):
        # Display input volume node in red slice background
        self.redSliceLogic.GetSliceCompositeNode().SetBackgroundVolumeID(volumeNode.GetID())
        self.redSliceLogic.FitSliceToAll()

    #------------------------------------------------------------------------------

#
# ChestXRayAnalyzerTest
#
class ChestXRayAnalyzerTest(ScriptedLoadableModuleTest):
    
    #-------------------------------------------------------------------------------
    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear()

    #-------------------------------------------------------------------------------
    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        self.test_ChestXRayAnalyzer1()

    #-------------------------------------------------------------------------------
    def test_ChestXRayAnalyzer1(self):
        pass
