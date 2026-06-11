import { StyleSheet } from 'react-native';

export const styles = StyleSheet.create({
    container: {
        flexDirection: 'row',
        backgroundColor: '#0a0a0f',
        height: 80,
        borderTopWidth: 0.5,
        borderTopColor: '#1e1e2e',
        paddingBottom: 20,
        position: 'absolute',
        bottom: 0,
        width: '100%',
    },
    tabItem: {
        flex: 1,
        alignItems: 'center',
        justifyContent: 'center',
    },
    label: {
        fontSize: 10,
        color: '#444468',
        marginTop: 4,
        fontWeight: '600',
    },
    activeText: {
        color: '#7c6cfa',
    },
});