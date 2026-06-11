import React from 'react';
import { styles } from './HomeScreen.styles';
import { View, Text, SafeAreaView, TouchableOpacity, StyleSheet, ScrollView } from 'react-native';
import { BottomNavigation } from '../components/BottomNavigaton';

export const HomeScreen = ({ navigation }: any) => {
  return (
    <SafeAreaView style={styles.container}>
      {/* 헤더 */}
      <View style={styles.header}>
        <Text style={styles.logo}>VeriFake</Text>
      </View>

      <ScrollView contentContainerStyle={styles.body}>
        {/* 서비스 소개(최근 기록 대신1) */}
        <View style={styles.introSection}>
          <Text style={styles.mainTitle}>조작된 영상으로부터{"\n"}진실을 확인하세요</Text>
          <Text style={styles.subTitle}>
            AI 기반 딥페이크 탐지 기술로 영상의 진위 여부와{"\n"}자연어 기반의 상세한 분석 근거를 제공합니다.
          </Text>
        </View>

        {/* 액션 버튼 */}
        <TouchableOpacity
          style={styles.mainBtn}
          onPress={() => navigation.navigate('DetectionInput')}
        >
          <View style={styles.btnContent}>
            <Text style={styles.btnIcon}>▶</Text>
            <Text style={styles.btnText}>새로운 영상 탐지 시작</Text>
          </View>
        </TouchableOpacity>

        {/* 가이드(최근 기록 대신2) */}
        <View style={styles.guideCard}>
          <Text style={styles.guideLabel}>TIP</Text>
          <Text style={styles.guideText}>
            SNS에서 확인하고 싶은 영상을{"\n"}
            '공유하기' 버튼을 통해 바로 분석할 수 있습니다.
          </Text>
        </View>
      </ScrollView>
      <BottomNavigation navigation={navigation} activeRoute="Home" />
    </SafeAreaView>
  );
};

