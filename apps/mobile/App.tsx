import './global.css';
import { NavigationContainer, createNavigationContainerRef } from '@react-navigation/native';
import { RootNavigator } from './src/navigation';
import { ShareIntentHandler } from './src/components/ShareIntentHandler';

export const navigationRef = createNavigationContainerRef();

export default function App() {
  return (
    <NavigationContainer ref={navigationRef}>
      <ShareIntentHandler />
      <RootNavigator />
    </NavigationContainer>
  );
}