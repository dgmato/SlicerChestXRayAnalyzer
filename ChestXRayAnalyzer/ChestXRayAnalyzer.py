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
import numpy as np
import skimage
        
        

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
        self.ui.modelComboBox.currentTextChanged.connect(self.updateParameterNodeFromGUI)

        # Buttons
        self.ui.runButton.connect('clicked(bool)', self.onRunButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

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
        self.ui.modelComboBox.setCurrentText(self._parameterNode.GetParameter("SelectedModel"))

        # Update buttons states and tooltips
        if self._parameterNode.GetNodeReference("InputVolume"):
            self.ui.runButton.toolTip = "Run automatic segmentation."
            self.ui.runButton.enabled = True
        else:
            self.ui.runButton.toolTip = "Select input volume."
            self.ui.runButton.enabled = False

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
        self._parameterNode.SetParameter("SelectedModel", self.ui.modelComboBox.currentText)

        self._parameterNode.EndModify(wasModified)

    #-------------------------------------------------------------------------------
    def onRunButton(self):
        """
        Run processing.
        """
        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):
            # Run inference
            self.logic.run(self._parameterNode)


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

        # Output segmentation
        self.outputSegmentation = None

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

        # Set red slice only layout
        slicer.app.layoutManager().setLayout(6)

    #-------------------------------------------------------------------------------
    def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """
        parameterNode.SetParameter("SelectedModel", "DynUNet Mandible and Nerves")

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

        # Get input volume
        try:
            inputVolumeNode = parameterNode.GetNodeReference("InputVolume")
        except:
            logging.error('Input volume was not defined.')

        # Crop input volume
        self.centerCropVolume(inputVolumeNode)

        # Automatic segmentation
        self.runAutomaticSegmentation(inputVolumeNode)

        # Classification
        self.runClassifier(inputVolumeNode)

        # Get final timestamp
        stopTime = time.time()

        # Report processing time
        logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')
        print(f'Processing completed in {stopTime-startTime:.2f} seconds')

        return True

    #-------------------------------------------------------------------------------
    def centerCropVolume(self, inputVolumeNode):
        # Get array
        volumeArray = slicer.util.arrayFromVolume(inputVolumeNode)

        # Crop center square of input image
        centerCropTransform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop()])
        volumeArray_centerCrop = centerCropTransform(volumeArray)

        # Update volume node with cropped array
        slicer.util.updateVolumeFromArray(inputVolumeNode, volumeArray_centerCrop)
    
    #-------------------------------------------------------------------------------
    def runAutomaticSegmentation(self, inputVolumeNode):

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
        print('Volume array = ', volumeArray.shape)

        # Save original image size
        numRows = volumeArray.shape[2]
        numColumns = volumeArray.shape[1]

        # Convert 8-bit image to [-1024, 1024] range
        img = xrv.datasets.normalize(volumeArray, volumeArray.max())

        # Resize image to 512 x 512
        transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(512)])
        img = transform(img)
        img = torch.from_numpy(img)

        # Get segmentation model
        model = xrv.baseline_models.chestx_det.PSPNet()

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
    def runClassifier(self, inputVolumeNode, progressDialog = None):

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

        # Resize image to 512 x 512
        transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(512)])
        img = transform(img)
        img = torch.from_numpy(img)

        # Get classification model
        model = xrv.models.ResNet(weights="resnet50-res512-all")

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
        if computeHeatmaps:
            heatmap_dict = {}
            heatmapProgressStep = 100/(len(model.pathologies))
            for pathologyId, pathologyName in enumerate(model.pathologies):
                print('Computing heatmap for pathology: ', pathologyName)
                # Report progress
                self.progress(progressDialog, heatmapProgressStep * (pathologyId+1), f'Running pathology classication...')
                # Enable gradient computation for this tensor
                inputImage = img[None,...]
                inputImage = inputImage.requires_grad_()
                # Run inference
                outputs = model(inputImage)
                #print(outputs.shape)                
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
        grads = torch.autograd.grad(outputs[:,target], image)[0][0][0]
        blurred = skimage.filters.gaussian(grads.detach().cpu().numpy()**2, sigma=(5, 5), truncate=3.5)
        return blurred

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
