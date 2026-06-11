import React from 'react';
import { View, Text, ScrollView, TouchableOpacity, Image } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { ArrowLeftIcon } from 'react-native-heroicons/outline';
import { styles } from './ResultScreen.styles';
import { BottomNavigation } from '../components/BottomNavigaton';
import type { MediaTaskResult } from '../api/verifakeApi';

interface ResultRouteParams {
    separatedMedia?: MediaTaskResult;
    thumbnailUri?: string | null;
}

function deriveIsFake(verdict: string | null | undefined): boolean | null {
    if (!verdict) return null;
    return verdict.toUpperCase() === 'FAKE';
}

export const ResultScreen = ({ navigation, route }: any) => {
    const { separatedMedia, thumbnailUri } = (route.params || {}) as ResultRouteParams;

    if (!separatedMedia) {
        return (
            <SafeAreaView style={styles.container}>
                <View style={styles.header}>
                    <TouchableOpacity onPress={() => navigation.navigate('Home')} style={styles.backBtn}>
                        <ArrowLeftIcon size={24} color="#7c6cfa" />
                    </TouchableOpacity>
                    <Text style={styles.headerTitle}>분석 결과</Text>
                    <View style={{ width: 24 }} />
                </View>
                <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
                    <Text style={{ color: '#a0a0ab', fontSize: 15 }}>분석 결과를 불러올 수 없습니다.</Text>
                    <TouchableOpacity onPress={() => navigation.navigate('Home')} style={{ marginTop: 20 }}>
                        <Text style={{ color: '#7c6cfa', fontSize: 15 }}>홈으로 돌아가기</Text>
                    </TouchableOpacity>
                </View>
                <BottomNavigation navigation={navigation} activeRoute="DetectionInput" />
            </SafeAreaView>
        );
    }

    const isFakeResult = deriveIsFake(separatedMedia.verdict);
    const isFake = isFakeResult ?? false;
    // 버그 수정: deepfakeScore → deepfake_score (백엔드 응답 키와 일치)
    const deepfakeScore = separatedMedia.deepfake_score ?? null;
    const hasVerdict = isFakeResult !== null;

    return (
        <SafeAreaView style={styles.container}>
            <View style={styles.header}>
                <TouchableOpacity onPress={() => navigation.navigate('Home')} style={styles.backBtn}>
                    <ArrowLeftIcon size={24} color="#7c6cfa" />
                </TouchableOpacity>
                <Text style={styles.headerTitle}>분석 결과</Text>
                <View style={{ width: 24 }} />
            </View>

            <ScrollView contentContainerStyle={styles.body}>
                {/* 썸네일 + 판정 배지 */}
                <View style={styles.mediaPreview}>
                    {thumbnailUri ? (
                        <Image source={{ uri: thumbnailUri }} style={styles.previewImage} resizeMode="cover" />
                    ) : (
                        <View style={[styles.previewImage, { backgroundColor: '#0a0a0f' }]} />
                    )}
                    {hasVerdict && (
                        <View style={{ position: 'absolute', top: 16, right: 16 }}>
                            <View style={[styles.statusBadge, isFake ? styles.bgFake : styles.bgReal]}>
                                <Text style={styles.statusBadgeText}>{isFake ? 'FAKE' : 'REAL'}</Text>
                            </View>
                        </View>
                    )}
                </View>

                {/* 딥페이크 가능성 수치 + 게이지 */}
                {deepfakeScore !== null ? (
                    <>
                        <Text style={styles.mainResultTitle}>딥페이크 가능성: {deepfakeScore}%</Text>
                        <View style={[styles.gaugeBar, { marginBottom: 24 }]}>
                            <View
                                style={[
                                    styles.gaugeFill,
                                    { width: `${deepfakeScore}%` },
                                    isFake ? styles.bgFake : styles.bgReal,
                                ]}
                            />
                        </View>
                    </>
                ) : (
                    <Text style={[styles.mainResultTitle, { marginBottom: 24 }]}>
                        {hasVerdict
                            ? `판정: ${isFake ? 'FAKE (조작 의심)' : 'REAL (정상)'}`
                            : '분석 결과를 가져오는 중...'}
                    </Text>
                )}

                {/* 분석 요약 카드 */}
                <View style={styles.anomalySection}>
                    <Text style={styles.anomalySubTitle}>
                        {hasVerdict
                            ? isFake
                                ? '이 영상에서 조작 의심 징후가 발견되었습니다.'
                                : '이 영상에서 조작 징후가 발견되지 않았습니다.'
                            : '분석이 완료되지 않았습니다.'}
                    </Text>
                    {separatedMedia.video_path ? (
                        <Text style={styles.anomalyText}>영상 파일이 정상적으로 처리되었습니다.</Text>
                    ) : null}
                    {separatedMedia.audio_path ? (
                        <Text style={styles.anomalyText}>음성 파일이 정상적으로 분리되었습니다.</Text>
                    ) : null}
                    {separatedMedia.error ? (
                        <Text style={styles.limitTitle}>! {separatedMedia.error}</Text>
                    ) : null}
                </View>

                {/* 상세 보고서 버튼 */}
                <TouchableOpacity
                    style={styles.primaryButton}
                    onPress={() => navigation.navigate('ResultDetail', { separatedMedia })}
                >
                    <Text style={styles.primaryButtonText}>상세 분석 보고서 확인하기</Text>
                </TouchableOpacity>

                <View style={{ height: 20 }} />
            </ScrollView>

            <BottomNavigation navigation={navigation} activeRoute="DetectionInput" />
        </SafeAreaView>
    );
};
