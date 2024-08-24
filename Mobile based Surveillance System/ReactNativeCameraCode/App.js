import React, { useState, useEffect, useRef } from 'react';
import { StyleSheet, Text, View, TouchableOpacity } from 'react-native';
import { Camera } from 'expo-camera';
import { Image } from 'expo-image';
import * as MediaLibrary from 'expo-media-library';
import * as ImageManipulator from 'expo-image-manipulator';


export default function CameraCheck() {
  const [uri,setUri]=useState('')
  const blurhash =
  '|rF?hV%2WCj[ayj[a|j[az_NaeWBj@ayfRayfQfQM{M|azj[azf6fQfQfQIpWXofj[ayj[j[fQayWCoeoeaya}j[ayfQa{oLj?j[WVj[ayayj[fQoff7azayj[ayj[j[ayofayayayj[fQj[ayayj[ayfjj[j[ayjuayj[';

  const [hasPermission, setHasPermission] = useState(null);
  const [type, setType] = useState(Camera.Constants.Type.back);
  const [isRecording, setIsRecording] = useState(false);
  const cameraRef = useRef(null);

  useEffect(() => {
    (async () => {
      const cameraStatus = await Camera.requestCameraPermissionsAsync();
      const audioStatus = await Camera.requestMicrophonePermissionsAsync();
      const mediaLibraryStatus = await MediaLibrary.requestPermissionsAsync();
      setHasPermission(
        cameraStatus.status === 'granted' &&
        audioStatus.status === 'granted' &&
        mediaLibraryStatus.status === 'granted'
      );
    })();
  }, []);

  if (hasPermission === null) {
    return <View />;
  }
  if (hasPermission === false) {
    return <Text>No access to camera</Text>;
  }

  const toggleCameraType = () => {
    setType(current => (current === Camera.Constants.Type.back ? Camera.Constants.Type.front : Camera.Constants.Type.back));
  };

  const saveMedia = async (uri) => {
    try {
      const asset = await MediaLibrary.createAssetAsync(uri);
      await MediaLibrary.createAlbumAsync('CameraApp', asset, false);
    } catch (error) {
      console.error('Error saving media to gallery:', error);
    }
  };

  const takePicture = async () => {
    if (cameraRef.current) {
      // const photo = await cameraRef.current.takePictureAsync();
      // console.log('Original URI:', photo.uri);
      //  console.log('1')
      // const manipulatedImage = await ImageManipulator.manipulateAsync(
      // photo.uri,
      // [{ resize: { width: 200, height: 200 } }],
      // { format: 'png', base64: true });

      //  console.log(manipulatedImage.uri)
      
      //  setUri(manipulatedImage.uri)
      // console.log('ok')
      
      console.log('http://192.168.233.13//getResponse.php')
      const response=await fetch('http://192.168.233.13/getResponse.php')
      const res=await response.json()
      console.log(res)
      
console.log('ok')
// let result = await ImagePicker.launchImageLibraryAsync({
//   mediaTypes: ImagePicker.MediaTypeOptions.All,
//   allowsEditing: true,
//   base64: true, //<-- boolean base64
//   aspect: [4, 3],
//   quality: 1,
// });

// console.log(result);
      // const image = Asset.fromModule(require(manipulatedImage.uri));
      //  console.log('1')
      // await image.downloadAsync();
      //  console.log('1')
      // console.log(image)

      // console.log(photo)
      // saveMedia(photo.uri);
      // console.log('1')
    //   const manipulatedImage = await ImageManipulator.manipulateAsync(
    //   photo.uri,
    //   [],
    //   { format: 'png', base64: true } // Set the format and base64 to true
    // );
console.log('2')
    // The manipulatedImage.uri now contains the matrix representation of the image

    // You can access the base64 data directly or convert it to a Uint8Array
    // const base64Data = manipulatedImage.base64;
    // const uint8Array = new Uint8Array(Buffer.from(base64Data, 'base64'));
console.log('3')
    // Now you can use the uint8Array as needed

    // Optionally, you can log or process the image data
    // console.log('Base64 Data:', base64Data);
    // console.log('Uint8Array:', uint8Array);

    // If you need the original URI for any reason, you can still access it from photo.uri
    // console.log('Original URI:', photo.uri);
    }
  };

  const handleRecord = async () => {
    if (isRecording) {
      cameraRef.current.stopRecording();
    } else {
      if (cameraRef.current) {
        try {
          setIsRecording(true);
          const video = await cameraRef.current.recordAsync();
          setIsRecording(false);
          saveMedia(video.uri);
        } catch (error) {
          console.error("Error during video recording:", error);
          setIsRecording(false);
        }
      }
    }
  };
  return (
    <View style={styles.container}>
      <Camera style={styles.camera} type={type} ref={cameraRef}>
        <View style={styles.buttonContainer}>
          <TouchableOpacity style={styles.button} onPress={toggleCameraType}>
            <Text style={styles.text}>Flip</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.button} onPress={takePicture}>
            <Text style={styles.text}>Take Photo</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.button} onPress={handleRecord}>
            <Text style={styles.text}>{isRecording ? 'Stop Recording' : 'Start Recording'}</Text>
          </TouchableOpacity>
          
        </View>
      </Camera>
      <Image
        style={{flex:2}}
        source={uri}
        placeholder={blurhash}
        contentFit="cover"
        transition={1000}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
  },
  camera: {
    flex: 1,
  },
  buttonContainer: {
    flex: 1,
    flexDirection: 'row',
    margin: 20,
    backgroundColor: 'transparent',
    justifyContent: 'center',
    alignItems: 'flex-end',
  },
  button: {
    padding: 13,
    backgroundColor: 'gray',
    margin: 10,

  },
  text: {
    fontSize: 18,
    color: 'white',
  },
});