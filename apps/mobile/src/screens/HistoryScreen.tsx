import React, { useEffect, useState } from 'react';
import { View, Text, SafeAreaView, FlatList, Image, TouchableOpacity, ActivityIndicator } from 'react-native';
import { styles } from './HistoryScreen.styles';
import { BottomNavigation } from '../components/BottomNavigaton';
import { getHistory } from '../api/verifakeApi';
import type { HistoryItem } from '../api/verifakeApi';

export const HistoryScreen = ({ navigation }: any) => {
    const [historyData, setHistoryData] = useState<HistoryItem[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        let isMounted = true;

        async function fetchHistory() {
            try {
                setLoading(true);
                setError(null);
                const data = await getHistory();
                if (isMounted) setHistoryData(data);
            } catch (e: any) {
                if (isMounted) setError(e.message ?? '기록을 불러오지 못했습니다.');
            } finally {
                if (isMounted) setLoading(false);
            }
        }

        fetchHistory();
        return () => { isMounted = false; };
    }, []);

    const renderItem = ({ item }: { item: HistoryItem }) => {
        // 버그 수정: item.status는 'COMPLETED'/'FAILED' 등이므로 verdict로 판별
        const isFake = item.verdict?.toUpperCase() === 'FAKE';

        return (
            <TouchableOpacity
                style={styles.historyCard}
                // 버그 수정: separatedMedia 파라미터 전달
                onPress={() => navigation.navigate('Result', {
                    separatedMedia: {
                        task_id: item.task_id,
                        status: item.status,
                        verdict: item.verdict,
                        deepfake_score: item.score,
                    },
                    thumbnailUri: item.thumb ?? null,
                })}
            >
                <Image
                    source={{ uri: item.thumb ?? 'https://via.placeholder.com/100' }}
                    style={styles.thumbnail}
                />

                <View style={styles.infoContainer}>
                    <View style={[styles.statusBadge, isFake ? styles.bgFake : styles.bgReal]}>
                        {/* 버그 수정: verdict를 badge에 표시 (status가 아닌 FAKE/REAL) */}
                        <Text style={styles.statusText}>
                            {item.verdict ? item.verdict.toUpperCase() : item.status}
                        </Text>
                    </View>
                    <Text style={styles.videoTitle} numberOfLines={1}>{item.title}</Text>
                    <Text style={styles.dateText}>{item.date}</Text>
                </View>

                <View style={styles.scoreContainer}>
                    <Text style={[styles.scoreValue, isFake ? styles.textFake : styles.textReal]}>
                        {item.score !== null ? `${item.score}%` : '-'}
                    </Text>
                </View>
            </TouchableOpacity>
        );
    };

    const renderContent = () => {
        if (loading) {
            return (
                <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
                    <ActivityIndicator size="large" color="#7c6cfa" />
                    <Text style={{ color: '#a0a0ab', marginTop: 12, fontSize: 14 }}>기록을 불러오는 중...</Text>
                </View>
            );
        }

        if (error) {
            return (
                <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center', paddingHorizontal: 32 }}>
                    <Text style={{ color: '#ff453a', fontSize: 15, textAlign: 'center' }}>{error}</Text>
                    <Text style={{ color: '#a0a0ab', marginTop: 8, fontSize: 13, textAlign: 'center' }}>
                        네트워크 연결을 확인하고 다시 시도해주세요.
                    </Text>
                </View>
            );
        }

        if (historyData.length === 0) {
            return (
                <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
                    <Text style={{ color: '#a0a0ab', fontSize: 15 }}>탐지 기록이 없습니다.</Text>
                    <Text style={{ color: '#666680', marginTop: 8, fontSize: 13 }}>
                        영상을 분석하면 여기에 기록이 남아요.
                    </Text>
                </View>
            );
        }

        return (
            <FlatList
                data={historyData}
                renderItem={renderItem}
                // 버그 수정: item.id는 이미 String으로 변환된 상태
                keyExtractor={(item) => item.id}
                contentContainerStyle={styles.listContent}
                showsVerticalScrollIndicator={false}
            />
        );
    };

    return (
        <SafeAreaView style={styles.container}>
            <View style={styles.header}>
                <Text style={styles.headerTitle}>전체 탐지 기록</Text>
            </View>

            {renderContent()}

            <BottomNavigation activeRoute="History" />
        </SafeAreaView>
    );
};
