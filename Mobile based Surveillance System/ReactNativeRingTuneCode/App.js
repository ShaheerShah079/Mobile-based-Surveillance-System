import React,{useState,useEffect} from 'react';
import { View, Button } from 'react-native';
import { Audio } from 'expo-av';


const RingtonePlayer = () => {

  const [ss,Setss]=useState(true)
  
    useEffect(() => {
      if(ss){
    const intervalId = setInterval(playRingtone, 4000);

    return () => clearInterval(intervalId);
    }
  }, [ss]); 

  const playRingtone = async () => {
    const { sound } = await Audio.Sound.createAsync(
      require('./assets/ring1.mp3')
    );
    await sound.playAsync();
  };

  return (
    <View style={{marginTop:200}}>
      <Button title="Play Ringtone" onPress={()=>{
        Setss(true);
        playRingtone()}} />
      <Button title="Stop" onPress={()=>{
        console.log('false')
        Setss(false)}} />
    </View>
  );
};
export default RingtonePlayer;
