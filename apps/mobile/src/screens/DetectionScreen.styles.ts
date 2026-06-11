import { StyleSheet } from 'react-native';

export const styles = StyleSheet.create({
    container: { flex: 1, backgroundColor: '#0a0a0f' },
    content: { flex: 1, paddingHorizontal: 24, paddingTop: 40 },
    title: { color: '#fff', fontSize: 24, fontWeight: 'bold', marginBottom: 8 },
    subTitle: { color: '#444468', fontSize: 14, marginBottom: 32 },

    // 업로드 박스
    uploadBox: {
        height: 200,
        backgroundColor: '#161622',
        borderRadius: 20,
        borderWidth: 2,
        borderColor: '#1e1e2e',
        borderStyle: 'dashed',
        justifyContent: 'center',
        alignItems: 'center',
        marginBottom: 24,
    },
    uploadText: { color: '#e1e1e6', fontSize: 16, fontWeight: '600', marginTop: 16 },
    uploadSubText: { color: '#444468', fontSize: 12, marginTop: 8 },

    divider: { flexDirection: 'row', alignItems: 'center', marginBottom: 24 },
    line: { flex: 1, height: 1, backgroundColor: '#1e1e2e' },
    dividerText: { color: '#444468', marginHorizontal: 16, fontSize: 12 },

    // 입력창
    inputContainer: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: '#11111d',
        borderRadius: 16,
        paddingHorizontal: 16,
        height: 56,
        borderWidth: 1,
        borderColor: '#1e1e2e',
        marginBottom: 20,
    },
    inputIcon: { marginRight: 12 },
    input: { flex: 1, color: '#fff', fontSize: 14 },

    // 시작 버튼
    startBtn: {
        backgroundColor: '#7c6cfa',
        height: 56,
        borderRadius: 16,
        justifyContent: 'center',
        alignItems: 'center',
        marginTop: 'auto',
        marginBottom: 100,
    },
    startBtnText: { color: '#fff', fontSize: 16, fontWeight: 'bold' },
});