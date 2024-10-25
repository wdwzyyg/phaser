import React from 'react';

export interface BoxProps {
    crossAlign?: 'center' | 'start' | 'end' | 'stretch'; // = 'center'

    children?: React.ReactNode;
}

export function HBox(props: BoxProps) {
    const style = {
        flexDirection: 'row',
        alignItems: props.crossAlign ?? 'center',
    } as const;

    return <div className="hbox" style={style}>
        { props.children }
    </div>;
}

export function VBox(props: BoxProps) {
    const style = {
        flexDirection: 'column',
        alignItems: props.crossAlign ?? 'center',
    } as const;

    return <div className="vbox" style={style}>
        { props.children }
    </div>;
}