import { StyleSheet } from 'react-native';

export const styles = StyleSheet.create({
    container: { flex: 1, backgroundColor: '#0a0a0f' },
    content: { flex: 1, paddingHorizontal: 30, justifyContent: 'center' },

    //로더
    loaderContainer: { alignItems: 'center', marginBottom: 60 },
    outerCircle: {
        width: 120,
        height: 120,
        borderRadius: 60,
        borderWidth: 2,
        borderColor: '#7c6cfa',
        justifyContent: 'center',
        alignItems: 'center',
        marginBottom: 24,
    },
    innerCircle: {
        width: 80,
        height: 80,
        borderRadius: 40,
        backgroundColor: '#1a1a2e',
        justifyContent: 'center',
        alignItems: 'center',
    },
    icon: { fontSize: 30 },
    title: { color: '#fff', fontSize: 20, fontWeight: 'bold', marginBottom: 8 },
    subTitle: { color: '#444468', fontSize: 14 },

    // 리스트
    stepList: { gap: 12 },
    stepItem: {
        flexDirection: 'row',
        alignItems: 'center',
        padding: 16,
        borderRadius: 14,
        borderWidth: 1,
    },
    stepDone: { backgroundColor: '#101a14', borderColor: '#1d3d2a' },
    stepWait: { backgroundColor: '#11111d', borderColor: '#1e1e2e' },

    checkCircle: {
        width: 20, height: 20, borderRadius: 10,
        borderWidth: 1, borderColor: '#444468',
        marginRight: 12, justifyContent: 'center', alignItems: 'center'
    },
    checkCircleDone: { backgroundColor: '#34c759', borderColor: '#34c759' },
    checkCircleError: { backgroundColor: '#ff453a', borderColor: '#ff453a' },
    stepLabel: { color: '#444468', fontSize: 15 },
    textDone: { color: '#34c759', fontWeight: '600' },
    textError: { color: '#ff453a', fontWeight: '600' },
});

export const iconProps = {
    checkIcon: { size: 12, color: '#fff', strokeWidth: 3 },
    loadingIcon: { size: 16, color: '#7c6cfa', strokeWidth: 2.5 },
};
