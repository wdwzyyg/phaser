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

interface SectionProps {
    name: string
    children?: React.ReactNode
}

export function Section(props: SectionProps) {
    const [collapsed, setCollapsed] = React.useState(false);

    function toggle(e: React.MouseEvent) {
        e.stopPropagation();
        setCollapsed(!collapsed);
    }

    return <>
        <div className="section-header" onClick={toggle}>{ props.name }</div>
        <div className={ "section" + (collapsed ? " collapsed" : "") }>{ props.children }</div>
    </>;
}