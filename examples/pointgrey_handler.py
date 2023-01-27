# coding=utf-8
# =============================================================================
# Copyright (c) 2001-2021 FLIR Systems, Inc. All Rights Reserved.
#
# This software is the confidential and proprietary information of FLIR
# Integrated Imaging Solutions, Inc. ("Confidential Information"). You
# shall not disclose such Confidential Information and shall use it only in
# accordance with the terms of the license agreement you entered into
# with FLIR Integrated Imaging Solutions, Inc. (FLIR).
#
# FLIR MAKES NO REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE
# SOFTWARE, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE, OR NON-INFRINGEMENT. FLIR SHALL NOT BE LIABLE FOR ANY DAMAGES
# SUFFERED BY LICENSEE AS A RESULT OF USING, MODIFYING OR DISTRIBUTING
# THIS SOFTWARE OR ITS DERIVATIVES.
# =============================================================================
#
# This AcquireAndDisplay.py shows how to get the image data, and then display images in a GUI.
# This example relies on information provided in the ImageChannelStatistics.py example.
#
# This example demonstrates how to display images represented as numpy arrays.
# Currently, this program is limited to single camera use.
# NOTE: keyboard and matplotlib must be installed on Python interpreter prior to running this example.

#import matplotlib.pyplot as plt; plt.ion()
import sys
import keyboard
import PySpin
import numpy as np

class FakeCameraHandler:

    def __init__(self,offset_x=900,offset_y=900,width=200,height=200,exposure=10000,gain=0):
        self._cam_offset_x  = offset_x
        self._cam_offset_y  = offset_y
        self._cam_width     = width
        self._cam_height    = height
        self._exposure_time = exposure
        self._gain          = gain
    
    class FrameGrabber:
        def __init__(self,cam_obj):
            self._cam_obj = cam_obj

        def __enter__(self):
            return self

        def grab(self,nframes=1):
            frames = []
            for _ in range(nframes):
                frames.append(np.random.rand(*[self._cam_obj._cam_width,self._cam_obj._cam_height])*2**16)
            return frames

        def __exit__(self,*args):
            pass

class CameraHandler:
    
    continue_recording = True

    def __init__(self,offset_x=900,offset_y=900,width=200,height=200,exposure=10000,gain=0):
        self._cam_offset_x  = offset_x
        self._cam_offset_y  = offset_y
        self._cam_width     = width
        self._cam_height    = height
        self._exposure_time = exposure
        self._gain          = gain

    def handle_close(self,evt):
        """
        This function will close the GUI when close event happens.

        :param evt: Event that occurs when the figure closes.
        :type evt: Event
        """
        self.continue_recording = False

    def configure_camera(self, nodemap):
        """
        This function sets the camera configuration.

        :param nodemap: Device nodemap.

        :return: True if successful, False otherwise.
        :rtype: bool
        """
        node_exposure_auto = PySpin.CEnumerationPtr(nodemap.GetNode('ExposureAuto'))
        if not PySpin.IsReadable(node_exposure_auto) or not PySpin.IsWritable(node_exposure_auto):
            return False

        exposure_auto_off = node_exposure_auto.GetEntryByName('Off')
        if not PySpin.IsReadable(exposure_auto_off):
            return False

        node_exposure_auto.SetIntValue(exposure_auto_off.GetValue())

        node_gain_auto = PySpin.CEnumerationPtr(nodemap.GetNode('GainAuto'))
        if not PySpin.IsReadable(node_gain_auto) or not PySpin.IsWritable(node_gain_auto):
            return False

        gain_auto_off = node_gain_auto.GetEntryByName('Off')
        if not PySpin.IsReadable(gain_auto_off):
            return False

        node_gain_auto.SetIntValue(gain_auto_off.GetValue())
        
        node_pixel_format = PySpin.CEnumerationPtr(nodemap.GetNode('PixelFormat'))
        if PySpin.IsReadable(node_pixel_format) and PySpin.IsWritable(node_pixel_format):
            # Retrieve the desired entry node from the enumeration node
            node_pixel_format_mono16 = PySpin.CEnumEntryPtr(node_pixel_format.GetEntryByName('Mono16'))
            if PySpin.IsReadable(node_pixel_format_mono16):
                # Retrieve the integer value from the entry node
                pixel_format_mono16 = node_pixel_format_mono16.GetValue()
                # Set integer as new value for enumeration node
                node_pixel_format.SetIntValue(pixel_format_mono16)
            else:
                print('Pixel format mono 16 not readable...')
        else:
            print('Pixel format not readable or writable...')

        node_width = PySpin.CIntegerPtr(nodemap.GetNode('Width'))
        if PySpin.IsReadable(node_width) and PySpin.IsWritable(node_width):
            node_width.SetValue(self._cam_width)
        else:
            print('Width not readable or writable...')

        node_height = PySpin.CIntegerPtr(nodemap.GetNode('Height'))
        if  PySpin.IsReadable(node_height) and PySpin.IsWritable(node_height):
            node_height.SetValue(self._cam_height)
        else:
            print('Height not readable or writable...')

        node_offset_x = PySpin.CIntegerPtr(nodemap.GetNode('OffsetX'))
        if PySpin.IsReadable(node_offset_x) and PySpin.IsWritable(node_offset_x):
            node_offset_x.SetValue(self._cam_offset_x)
        else:
            print('Offset X not readable or writable...')

        node_offset_y = PySpin.CIntegerPtr(nodemap.GetNode('OffsetY'))
        if PySpin.IsReadable(node_offset_y) and PySpin.IsWritable(node_offset_y):
            node_offset_y.SetValue(self._cam_offset_y)
        else:
            print('Offset Y not readable or writable...')
        
        # Set exposure time; exposure time recorded in microseconds
        node_exposure_time = PySpin.CFloatPtr(nodemap.GetNode('ExposureTime'))
        if not PySpin.IsReadable(node_exposure_time) or not PySpin.IsWritable(node_exposure_time):
            return False

        exposure_time_max = node_exposure_time.GetMax()

        if self._exposure_time > exposure_time_max:
            self._exposure_time = exposure_time_max

        node_exposure_time.SetValue(self._exposure_time)

        # Set gain; gain recorded in decibels
        node_gain = PySpin.CFloatPtr(nodemap.GetNode('Gain'))
        if not PySpin.IsReadable(node_gain) or not PySpin.IsWritable(node_gain):
            return False

        gain_max = node_gain.GetMax()

        if self._gain > gain_max:
            self._gain = gain_max

        node_gain.SetValue(self._gain)
        
        return True


    def acquire_images(self, cam, nodemap, nodemap_tldevice):
        sNodemap = cam.GetTLStreamNodeMap()

        # Change bufferhandling mode to NewestOnly
        node_bufferhandling_mode = PySpin.CEnumerationPtr(sNodemap.GetNode('StreamBufferHandlingMode'))
        if not PySpin.IsReadable(node_bufferhandling_mode) or not PySpin.IsWritable(node_bufferhandling_mode):
            print('Unable to set stream buffer handling mode.. Aborting...')
            return False

        # Retrieve entry node from enumeration node
        node_newestonly = node_bufferhandling_mode.GetEntryByName('NewestOnly')
        if not PySpin.IsReadable(node_newestonly):
            print('Unable to set stream buffer handling mode.. Aborting...')
            return False

        # Retrieve integer value from entry node
        node_newestonly_mode = node_newestonly.GetValue()

        # Set integer value from entry node as new value of enumeration node
        node_bufferhandling_mode.SetIntValue(node_newestonly_mode)

        print('*** IMAGE ACQUISITION ***\n')
        try:
            node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
            if not PySpin.IsReadable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
                print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
                return False

            # Retrieve entry node from enumeration node
            node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
            if not PySpin.IsReadable(node_acquisition_mode_continuous):
                print('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
                return False

            # Retrieve integer value from entry node
            acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

            # Set integer value from entry node as new value of enumeration node
            node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

            print('Acquisition mode set to continuous...')

            #  Begin acquiring images
            #
            #  *** NOTES ***
            #  What happens when the camera begins acquiring images depends on the
            #  acquisition mode. Single frame captures only a single image, multi
            #  frame catures a set number of images, and continuous captures a
            #  continuous stream of images.
            #
            #  *** LATER ***
            #  Image acquisition must be ended when no more images are needed.
            cam.BeginAcquisition()

            print('Acquiring images...')

            #  Retrieve device serial number for filename
            #
            #  *** NOTES ***
            #  The device serial number is retrieved in order to keep cameras from
            #  overwriting one another. Grabbing image IDs could also accomplish
            #  this.
            device_serial_number = ''
            node_device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
            if PySpin.IsReadable(node_device_serial_number):
                device_serial_number = node_device_serial_number.GetValue()
                print('Device serial number retrieved as %s...' % device_serial_number)

            # Close program
            print('Press enter to close the program..')

            # Retrieve and display images
            while(self.continue_recording):
                try:

                    #  Retrieve next received image
                    #
                    #  *** NOTES ***
                    #  Capturing an image houses images on the camera buffer. Trying
                    #  to capture an image that does not exist will hang the camera.
                    #
                    #  *** LATER ***
                    #  Once an image from the buffer is saved and/or no longer
                    #  needed, the image must be released in order to keep the
                    #  buffer from filling up.
                    
                    image_result = cam.GetNextImage(1000)

                    #  Ensure image completion
                    if image_result.IsIncomplete():
                        print('Image incomplete with image status %d ...' % image_result.GetImageStatus())

                    else:                    

                        # Getting the image data as a numpy array
                        image_data = image_result.GetNDArray()
                        
                        if self._callback_function is not None:
                            # Pass image array to callback function
                            self._callback_function(image_data)
                        
                        # If user presses enter, close the program
                        if keyboard.is_pressed('ENTER'):
                            print('Program is closing...')
                            input('Done! Press Enter to exit...')
                            self.continue_recording=False                        

                    #  Release image
                    #
                    #  *** NOTES ***
                    #  Images retrieved directly from the camera (i.e. non-converted
                    #  images) need to be released in order to keep from filling the
                    #  buffer.
                    image_result.Release()

                except PySpin.SpinnakerException as ex:
                    print('Error: %s' % ex)
                    return False

            #  End acquisition
            #
            #  *** NOTES ***
            #  Ending acquisition appropriately helps ensure that devices clean up
            #  properly and do not need to be power-cycled to maintain integrity.
            cam.EndAcquisition()

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            return False

        return True

    def print_device_info(self, nodemap):
        """
        This function prints the device information of the camera from the transport
        layer; please see NodeMapInfo example for more in-depth comments on printing
        device information from the nodemap.

        :param nodemap: Transport layer device nodemap.
        :type nodemap: INodeMap
        :returns: True if successful, False otherwise.
        :rtype: bool
        """

        print('*** DEVICE INFORMATION ***\n')

        try:
            result = True
            node_device_information = PySpin.CCategoryPtr(nodemap.GetNode('DeviceInformation'))

            if PySpin.IsReadable(node_device_information):
                features = node_device_information.GetFeatures()
                for feature in features:
                    node_feature = PySpin.CValuePtr(feature)
                    print('%s: %s' % (node_feature.GetName(),
                                    node_feature.ToString() if PySpin.IsReadable(node_feature) else 'Node not readable'))

            else:
                print('Device control information not readable.')

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            return False

        return result

    def run_single_camera(self, cam):
        """
        This function acts as the body of the example; please see NodeMapInfo example
        for more in-depth comments on setting up cameras.

        :param cam: Camera to run on.
        :type cam: CameraPtr
        :return: True if successful, False otherwise.
        :rtype: bool
        """
        try:
            result = True

            nodemap_tldevice = cam.GetTLDeviceNodeMap()

            result &= self.print_device_info(nodemap_tldevice)
            
            # Initialize camera
            cam.Init()

            # Retrieve GenICam nodemap
            nodemap = cam.GetNodeMap()

            # Configure camera
            result &= self.configure_camera(nodemap)

            # Acquire images
            result &= self.acquire_images(cam, nodemap, nodemap_tldevice)

            # Deinitialize camera
            cam.DeInit()

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            result = False

        return result

    class FrameGrabber:
        def __init__(self,cam_obj):
            self._cam_obj = cam_obj

        def __enter__(self):

            # Retrieve singleton reference to system object
            self.system = PySpin.System.GetInstance()

            # Retrieve list of cameras from the system
            cam_list = self.system.GetCameras()
            num_cameras = cam_list.GetSize()

            # Finish if there are no cameras
            if num_cameras == 0:

                # Clear camera list before releasing system
                cam_list.Clear()

                # Release system instance
                self.system.ReleaseInstance()

                print('Not enough cameras!')
                input('Done! Press Enter to exit...')
            
            self.cam_list = cam_list
            cam = cam_list[0]
            self.cam = cam
            try:
                # Initialize camera
                cam.Init()

                # Retrieve GenICam nodemap
                nodemap = cam.GetNodeMap()

                # Configure camera
                self._cam_obj.configure_camera(nodemap)

            except PySpin.SpinnakerException as ex:
                print('Error: %s' % ex)

            sNodemap = cam.GetTLStreamNodeMap()

            # Change bufferhandling mode to NewestOnly
            node_bufferhandling_mode = PySpin.CEnumerationPtr(sNodemap.GetNode('StreamBufferHandlingMode'))
            if not PySpin.IsReadable(node_bufferhandling_mode) or not PySpin.IsWritable(node_bufferhandling_mode):
                print('Unable to set stream buffer handling mode.. Aborting...')
                return False

            # Retrieve entry node from enumeration node
            node_newestonly = node_bufferhandling_mode.GetEntryByName('NewestOnly')
            if not PySpin.IsReadable(node_newestonly):
                print('Unable to set stream buffer handling mode.. Aborting...')
                return False

            # Retrieve integer value from entry node
            node_newestonly_mode = node_newestonly.GetValue()

            # Set integer value from entry node as new value of enumeration node
            node_bufferhandling_mode.SetIntValue(node_newestonly_mode)

            node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
            if not PySpin.IsReadable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
                print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
                return False

            # Retrieve entry node from enumeration node
            node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
            if not PySpin.IsReadable(node_acquisition_mode_continuous):
                print('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
                return False

            # Retrieve integer value from entry node
            acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

            # Set integer value from entry node as new value of enumeration node
            node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

            return self

        def grab(self,nframes=1):
            frames = []
            self.cam.BeginAcquisition()
            # Retrieve and display images
            for _ in range(nframes):
                image_result = self.cam.GetNextImage(1000)
                #  Ensure image completion
                if image_result.IsIncomplete():
                    print('Image incomplete with image status %d ...' % image_result.GetImageStatus())
                else:                    
                    # Getting the image data as a numpy array
                    frames.append(image_result.GetNDArray().copy())
                image_result.Release()
            self.cam.EndAcquisition()
            return frames

        def __exit__(self,*args):
            # Deinitialize camera
            self.cam_list[0].DeInit()
            del self.cam 

            # Clear camera list before releasing system
            self.cam_list.Clear()

            # Release system instance
            self.system.ReleaseInstance()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.ion()
    cam = CameraHandler(offset_x=200,offset_y=300,width=40,height=40)
    with cam.FrameGrabber(cam) as fg:
        plt.matshow(fg.grab()[0])
