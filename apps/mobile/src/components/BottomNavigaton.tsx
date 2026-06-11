import React from 'react';
import { View, Text, TouchableOpacity } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import {
    HomeIcon,
    MagnifyingGlassIcon,
    ClipboardDocumentListIcon
} from 'react-native-heroicons/outline';
import { styles } from './BottomNavigation.styles';

export const BottomNavigation = ({ activeRoute }: any) => {
    const navigation = useNavigation<any>();
    const tabs = [
        { name: 'Home', label: '홈', Icon: HomeIcon },
        { name: 'DetectionInput', label: '탐지', Icon: MagnifyingGlassIcon },
        { name: 'History', label: '기록', Icon: ClipboardDocumentListIcon },
    ];

    return (
        <View style={styles.container}>
            {tabs.map((tab) => {
                const isActive = activeRoute === tab.name;
                const CurrentIcon = tab.Icon;

                return (
                    <TouchableOpacity
                        key={tab.name}
                        style={styles.tabItem}
                        onPress={() => { navigation.navigate(tab.name) }}
                    >
                        <CurrentIcon
                            size={24}
                            color={isActive ? '#7c6cfa' : '#444468'}
                        />
                        <Text style={[styles.label, isActive && styles.activeText]}>
                            {tab.label}
                        </Text>
                    </TouchableOpacity>
                );
            })}
        </View>
    );
};