import { StyleSheet, Dimensions } from 'react-native';

const { width } = Dimensions.get('window');

export const styles = StyleSheet.create({
    container: { flex: 1, backgroundColor: '#0a0a0f' },
    body: { paddingHorizontal: 24, paddingBottom: 100 },

    headerRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginTop: 40,
        marginBottom: 30
    },
    welcomeText: { color: '#7c6cfa', fontSize: 16, fontWeight: '500' },
    headerTitle: { color: '#fff', fontSize: 28, fontWeight: 'bold', marginTop: 4 },

    // 버튼
    editFloatingBtn: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: '#7c6cfa',
        paddingHorizontal: 16,
        paddingVertical: 10,
        borderRadius: 25,
        elevation: 8,
        shadowColor: '#7c6cfa',
        shadowOpacity: 0.4,
        shadowRadius: 10,
    },
    editFloatingText: { color: '#fff', fontSize: 14, fontWeight: 'bold', marginLeft: 6 },

    actionButtons: { flexDirection: 'row', gap: 12 },
    circleBtnCancel: { backgroundColor: '#1e1e2e', padding: 10, borderRadius: 20 },
    circleBtnSave: { backgroundColor: '#1e1e2e', padding: 10, borderRadius: 20 },

    // 카드
    infoCard: {
        backgroundColor: '#161622',
        borderRadius: 28,
        padding: 24,
        borderWidth: 1,
        borderColor: 'rgba(124, 108, 250, 0.1)',
        elevation: 4,
    },
    cardHeader: { color: '#444468', fontSize: 13, fontWeight: 'bold', marginBottom: 25, letterSpacing: 1 },

    fieldItem: { flexDirection: 'row', alignItems: 'center', marginVertical: 12 },
    iconCircle: {
        width: 40,
        height: 40,
        backgroundColor: '#1e1e2e',
        borderRadius: 12,
        justifyContent: 'center',
        alignItems: 'center',
        marginRight: 16
    },
    fieldTextContainer: { flex: 1 },
    fieldLabel: { color: '#444468', fontSize: 12, marginBottom: 4 },
    fieldValue: { color: '#fff', fontSize: 17, fontWeight: '600' },
    readOnlyValue: { color: '#777', fontSize: 16 },

    // 입력창 활성화 시 강조
    activeInput: {
        color: '#fff',
        fontSize: 17,
        fontWeight: '600',
        borderBottomWidth: 2,
        borderBottomColor: '#7c6cfa',
        paddingVertical: 4,
    },

    separator: {
        height: 1,
        backgroundColor: 'rgba(255,255,255,0.05)',
        marginVertical: 10,
        marginLeft: 56
    },

    accountBadge: {
        alignSelf: 'center',
        marginTop: 40,
        backgroundColor: 'rgba(124, 108, 250, 0.1)',
        paddingHorizontal: 15,
        paddingVertical: 6,
        borderRadius: 20,
        borderWidth: 1,
        borderColor: 'rgba(124, 108, 250, 0.2)',
    },
    badgeText: { color: '#7c6cfa', fontSize: 12, fontWeight: 'bold' }
});