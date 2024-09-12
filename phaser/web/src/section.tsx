import React from 'react';

interface SectionProps {
    name: string
    children?: React.ReactNode
}

export default function Section(props: SectionProps) {
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