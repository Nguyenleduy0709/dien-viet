def dress_svg(visual: str) -> str:
    visual = visual or "Áo dài"

    return f"""
    <svg
        xmlns="http://www.w3.org/2000/svg"
        width="600"
        height="800"
        viewBox="0 0 600 800"
    >
        <rect
            width="600"
            height="800"
            fill="#f5f5f5"
        />

        <text
            x="300"
            y="350"
            text-anchor="middle"
            font-size="32"
            font-family="Arial"
            fill="#333"
        >
            {visual}
        </text>

        <text
            x="300"
            y="410"
            text-anchor="middle"
            font-size="22"
            font-family="Arial"
            fill="#777"
        >
            Diện Việt
        </text>
    </svg>
    """