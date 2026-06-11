import { StyleSheet } from 'react-native';

export const styles = StyleSheet.create({
    container: { flex: 1, backgroundColor: '#0a0a0f' },
    header: {
        paddingHorizontal: 20,
        paddingVertical: 20,
        borderBottomWidth: 1,
        borderBottomColor: '#1e1e2e',
    },
    headerTitle: { color: '#fff', fontSize: 22, fontWeight: 'bold' },
    listContent: { paddingHorizontal: 20, paddingTop: 20, paddingBottom: 100 },

    // 기록 카드
    historyCard: {
        flexDirection: 'row',
        backgroundColor: '#161622',
        borderRadius: 18,
        padding: 12,
        marginBottom: 16,
        borderWidth: 1,
        borderColor: '#1e1e2e',
    },
    thumbnail: {
        width: 80,
        height: 80,
        borderRadius: 12,
        backgroundColor: '#252538',
    },
    infoContainer: {
        flex: 1,
        marginLeft: 16,
        justifyContent: 'center',
    },
    statusBadge: {
        alignSelf: 'flex-start',
        paddingHorizontal: 8,
        paddingVertical: 4,
        borderRadius: 6,
        marginBottom: 6,
    },
    statusText: { color: '#fff', fontSize: 10, fontWeight: 'bold' },
    videoTitle: { color: '#fff', fontSize: 15, fontWeight: '600', marginBottom: 4 },
    dateText: { color: '#444468', fontSize: 12 },

    // 점수/결과 표시
    scoreContainer: {
        justifyContent: 'center',
        alignItems: 'flex-end',
        paddingLeft: 10,
    },
    scoreValue: { fontSize: 18, fontWeight: 'bold' },
    textFake: { color: '#ff453a' },
    textReal: { color: '#32d74b' },
    bgFake: { backgroundColor: '#ff453a' },
    bgReal: { backgroundColor: '#32d74b' },
});