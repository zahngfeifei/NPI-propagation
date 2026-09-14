from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import base64
import copy
import io
import math
import re
import shutil
import subprocess
import types
import xml.etree.ElementTree as ET
import zlib

from PIL import Image, ImageChops, ImageDraw, ImageFont

FIGURE_WIDTH_PIXELS = 1800
FIGURE_HEIGHT_PIXELS = 1320
ROW_SOURCE_WIDTH_PIXELS = 1800
REFERENCE_WIDTH_PIXELS = 1800
REFERENCE_HEIGHT_PIXELS = 1320
FINAL_FONT_FAMILY = "Arial"
FINAL_FONT_SIZE = 25
ROW_HEIGHT_PIXELS = (440, 420, 420, 380)
VISIBLE_MASK_THRESHOLD = 2


@dataclass(frozen=True)
class PixelRect:
    """Pixel rectangle stored consistently as left/top/right/bottom."""

    left: float
    top: float
    right: float
    bottom: float

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def center_x(self) -> float:
        return 0.5 * (self.left + self.right)

    @property
    def center_y(self) -> float:
        return 0.5 * (self.top + self.bottom)

    def as_crop_box(self) -> tuple[int, int, int, int]:
        return (
            round(self.left),
            round(self.top),
            round(self.right),
            round(self.bottom),
        )


@dataclass(frozen=True)
class LayoutTile:
    """One source crop and its destination slot in the final figure."""

    name: str
    row_index: int
    source: PixelRect
    slot: PixelRect
    horizontal_alignment: str = "center"
    vertical_alignment: str = "center"


# All final-image placement is declared here.  Source and destination rectangles
# use the same left/top/right/bottom convention, and every tile declares its own
# alignment instead of relying on tile-index special cases.
COMPOSITE_LAYOUT_TILES = (
    # Top row: keep text-bearing crops at exact 1:1 scale.
    LayoutTile(
        "top-a",
        0,
        PixelRect(0, 0, 980, 440),
        PixelRect(0, 0, 980, 440),
        horizontal_alignment="left",
        vertical_alignment="top",
    ),
    LayoutTile(
        "top-b",
        1,
        PixelRect(0, 0, 820, 420),
        PixelRect(980, 0, 1800, 420),
        horizontal_alignment="left",
        vertical_alignment="top",
    ),

    # Middle row: all plot crops use exact 1:1 source-to-destination geometry.
    # This prevents c/d/e typography from being shrunk or enlarged after rendering.
    LayoutTile(
        "middle-b",
        1,
        PixelRect(860, 0, 1670, 420),
        PixelRect(0, 415, 810, 835),
        horizontal_alignment="left",
        vertical_alignment="top",
    ),
    LayoutTile(
        "middle-c",
        2,
        PixelRect(0, 0, 330, 420),
        PixelRect(842, 415, 1172, 835),
        horizontal_alignment="left",
        vertical_alignment="top",
    ),
    LayoutTile(
        "middle-d",
        2,
        PixelRect(350, 0, 700, 420),
        PixelRect(1212, 415, 1562, 835),
        horizontal_alignment="left",
        vertical_alignment="top",
    ),
    # Panel e brain maps: crop away the embedded HC/ASD text, then
    # enlarge the brain-only tiles and place the entire label+brain block farther right.  The ASD tile is
    # placed lower to create a clear vertical gap between the two brain maps.
    LayoutTile(
        "middle-hc",
        3,
        PixelRect(100, 70, 235, 185),
        PixelRect(1560, 390, 1800, 605),
    ),
    LayoutTile(
        "middle-asd",
        3,
        PixelRect(100, 190, 235, 305),
        PixelRect(1560, 590, 1800, 805),
    ),

    # Bottom row: f/g/h plot crops stay at exact 1:1 scale.
    # Their brain-map slots are enlarged to 240 px width, matching panel e.
    LayoutTile(
        "bottom-f-plot",
        2,
        PixelRect(700, 0, 1050, 420),
        PixelRect(0, 900, 350, 1320),
        horizontal_alignment="left",
        vertical_alignment="top",
    ),
    LayoutTile(
        "bottom-f-hc",
        3,
        PixelRect(403, 70, 538, 185),
        PixelRect(380, 875, 620, 1090),
    ),
    LayoutTile(
        "bottom-f-asd",
        3,
        PixelRect(403, 190, 538, 305),
        PixelRect(380, 1075, 620, 1290),
    ),
    LayoutTile(
        "bottom-g-plot",
        2,
        PixelRect(1400, 0, 1800, 420),
        PixelRect(600, 900, 1000, 1320),
        horizontal_alignment="left",
        vertical_alignment="top",
    ),
    LayoutTile(
        "bottom-g-hc",
        3,
        PixelRect(1025, 70, 1160, 185),
        PixelRect(1035, 875, 1275, 1090),
    ),
    LayoutTile(
        "bottom-g-asd",
        3,
        PixelRect(1025, 190, 1160, 305),
        PixelRect(1035, 1075, 1275, 1290),
    ),
    LayoutTile(
        "bottom-h-plot",
        2,
        PixelRect(1050, 0, 1400, 420),
        PixelRect(1230, 900, 1580, 1320),
        horizontal_alignment="left",
        vertical_alignment="top",
    ),
    LayoutTile(
        "bottom-h-hc",
        3,
        PixelRect(715, 70, 850, 185),
        PixelRect(1560, 875, 1800, 1090),
    ),
    LayoutTile(
        "bottom-h-asd",
        3,
        PixelRect(715, 190, 850, 305),
        PixelRect(1560, 1075, 1800, 1290),
    ),
)

LAYOUT_TILE_BY_NAME = {tile.name: tile for tile in COMPOSITE_LAYOUT_TILES}


def getBrainGroupLabelPositions() -> tuple[tuple[float, float, str], ...]:
    """Brain-map HC/ASD annotations are intentionally omitted."""
    return ()


BOTTOM_ROW_HEADING_TOP = 900.0 + 17.4
BOTTOM_PANEL_LABEL_POSITIONS = (
    (8, BOTTOM_ROW_HEADING_TOP, "f"),
    (590, BOTTOM_ROW_HEADING_TOP, "g"),
    (1220, BOTTOM_ROW_HEADING_TOP, "h"),
)


ROW_1_SOURCE_B85 = (
    'c-rk;{cht(a{qsTc?Sk}3CUZTE6I|-5FH@f@{F+ZM`X|JZZrx(i7lBo6v>d3KgKYScQ_yjkRZ>H{}0H+1jyZMq^i0}Hk*{K+1m{c'
    'LG}(~k=<QYU0qfE>*{VU;&7$e_F|o`V`kf$w~E3z)tn#*Qz!MpAbI;{fpsD$z4g6WR^bXB-@XwK$?ZDz{QPm{q>&${SldaoRRksQ'
    'kF`#iq4$?d-K?o;Ey5tRS5Dw8S&Yr_`J6x2RgWI=7gmh?REQ`Q&qF_qlT6r=7ceKjX3G@|Qs$mu;mU~`P>__KFYvGLePLCK;bM`n'
    'bQV5jMEkjlj^S-XyQB)k6-b;59IRK-BMKQr`MbyoTzG?$yMoQtaFkJ{_6`2$<Ti|wLdndTg>irllYCTWvEv0v<jk4nI;jH!<em0I'
    '$F=96MYea?;7^eg&zbL&&{YE1)6@%=83BB@Z`sOAqFWX-VEN%<cs3lI+Gm#+H}>`L?+02>YaLk4w{Ncc7lV;K>L0_ua%sDbrTrg<'
    '*Y;?5F|fZMp5DC2R#yA%o15Xuck1ib+c&5Eo4)$`;O(2iczijop>llrLk*GRasTAI!A(v5v*EiBWAV<u8vZaCT~jxj&AjaWVEFF+'
    'ja1m|Lg7XK1`AIwKOBz+_Q~bw!}*0EbAC=FcCDh?+3=!Y*2O-Sj{Da)gE1#<k1s!5oDMJE+2{T7yWs`YX<E&eBI5Ak<oy7pc$xpV'
    'TDxm?wBKmeLHbs!mkEY#Joq>qd~cs#4S|7Cr<!lHmqa#!c#X8GCi6ZRwHkh%ha@QK@Z$9H+`b+RPSM1g?Y-uH`|$0X?*>2GSL4g8'
    '{=5Fo@bbb2l2^pE)zEeuTDPI?H?%{DK7;;v^rL<9Vf=BBS8AaZON~LlU~cFAANZ&=yKmpT1HrEB%kk-8j8ygaCwfEE``4!!(Rd-X'
    'LUK=7dZg*UYSMpMMd{7+o9iEe`MFH6WhX31!uS#Ezi-=4nnJ_^Z>8s5a_2bo>8WF%o?jqmK}Kd>&_RangZ+c!ZiOTkoptw5xO5VA'
    'YID+WwofER`0{jrx7ps;pL3q%@O3;tdJy3PSG$ImHGKOU`UOQNl`g_d{3CH4e5~g>YSl3twX0l4yJZHWjzs@raP$4;_`4boPuMYR'
    'c^$6K!&HB+vU}1`VdjAjQs1r}|05L9<FhKo6Lprq&m@87Y~ieZcwGp@EtPwGUB9mfC-w`rlhxR4Pg#vR3(Ly*lO)vO;%0m~JeB22'
    '9$`LRT{tUtq%Cym@!5tSFsnGeJ!i<lb9I>FW38y7d4VQ-M8WBLba^${k`2uB{-d4vVZ`i~-D+;Z<?S1nEi~x(!dn6eI?v~P!_;<u'
    '3z#5aM{nO~@ShWVj*rL(gup+Besc8b)AwHBhWE**PiLSz_(Qd<G+m%`=FN^n-`%e1%&jyYrhrQ3+co#()-n6XwHGrND`2|><4oR-'
    'Mp?9m%1jUe1dsqOkZDX14w7o=2Pn15$y;cEkf?&i9sp@2hB@UFp-3evR7^rUUc$69&u155dKRt&cM!*6Z0KXbL+hiQ1QE3bR7<pb'
    '04T(5fz_?7`zYW-aHthh;wk}7OaAS+E<PGb8XFlcGt&&f+89(7JFCQ6M*zW?@uXudYvHUspS+WVMTesIi4!C{35&f2lubn86iNYW'
    '-lps!CFTI+u<3lAq~S`JOXF3EX8ztw^Y=#{jbN#GwkV2cFy4hcfOes%f*PRQd0>f^dMPOQl}f%dWBv=39#U`q`PEvFTh;Q}lEDy@'
    'mNwVozWl26qS^xRFD)5RN%yy2%Ji&j^+kzu$Dr*?4SiU2+<R_%3q=Y5&Z;C4)BzQT5!5+z{DdhbvA4WU)v|Tq%|n;jD=%0lP<Rus'
    'nWPu6MMgnCYn0qANn7b-glf<O2xur}<Syj*o%XX?1j98LdK^a1k`@d7ICdTrW4?~>SaJ#;G*4loIigTRLDCD-qb%aUN;1%r^~x|>'
    '<bXDMp3>A>`r*vbe`9~N9pATt=utP#oV_2;!IOqTwB}xFfGhfsiC4?o_2%gWoU=i%rb3J;4u1z#$05Xoo>pmZ!BPn3p=8n)5y?q?'
    'CpiJg;-M2`h=ZvV)`o^riF6WaE9U*Up5Hob!Qytabzpaz%_~}HCHDRWOYg(@b7F_F3$Y}9NOcpVpWZ(no(@_%(!;<KQBX1S=!l_A'
    '5owzMnaQyOGH5@taA#j5chXX4<}>XcSE7|5sSHg?36pJl&k5KOi|u4R!_rbG?LYqWfBfkWzyIUE|C`cZLWmnojaT@s)|{4fvQgG;'
    'w8O>GK+YW@C-x>|ENGKZWR)-*q9~i@7DW_L*2;M>HVW9#KC{POeQ)#0IsxF<kx5nz7uG@*wRUuT`z{XG(RJ#i31Bsf4FCxV&N)eH'
    '2jST%{Bw;Fs^q9o`iiBoH^)e_6*nZhnmH*k1z?EaP>?{aWhRh_YSuitqsb=rn7eX<ZQ1M+eXf#Sd*Q{<2^bLX4j^Pk3qWO(GswCM'
    '3@VB&B$c!T>oNNdlttK<6lO~rQ^}q>sq-_DwlP&>uM_5GOp<jJ`3$pD4k7xl<$_4cXZbLc$YnezoQ!C4#W5%cmq`;Lh+@1@O?i`}'
    '<(N4x#pfFgEQ#UevHD02$V!loTnklshSO9QtSLHu#~ult=JqHXjXb{Upj7mybheCQa|Bz6JOR&~Qjmim7@-PzMA%X4Zicg4L)puK'
    'X~1w5LTt%UY3>+YfQ^yID-@5bA&}?@)rw<c!(bbVwC2XH;k&=;m9_O;;V~(dPl4qIX61SfEn!hl|MV&O21eu->a)0qiz%}de<Vi5'
    '^zXQ2=2D=<kv4>7Q0=3VB{j6o*d$$pZA=Oa3?eIVDYk=*uf7IkGnGI^DR5`D^h9f%0OSC@tU{1uy;yh;hAzXKBds!*ZUKs(rOAf+'
    '{W?z(C^&KQ{;%O((^|*?Yl-<SbI+wgz)4Sfv;eqG<QEh0%$W^q{i)@Fdl<C}G_}9cUTUQD-JDwUFnTnE8L@{5dce5BoZ$4Ds8TJl'
    'X!dCEDN;9S&10B=yKNljjFRY|La5s5ZRufDsFDPKs}Z>pGCn%~nu%C(wE?gQeBy>28i5VOT6)Xi7pzNJ$}%l*r_aMjlCo$Bfqzcn'
    '1I*q@h>}s8KLuaF1ws$}0K7?SaT>I3aJ^oUC6YdPHU_ADVR4ukhR=d}f@VVlbZTaF7vSwF4^cTCv`;D7D*M2TJS%zKhK6Fu`5yBp'
    'FkY4{LS_w0F{)!c$!kov@7PJ`uU7%hOm$hmNs+>uzLL_K7P*g2j3slUD!DV)SUdT2_5DOxKg<Wcg3naWr+|>$YKJcJFE|w*E(uZ>'
    'V7Yzbd{7@cCOHF$*{hWk*(+EX_w+ck_3S_JMU{Y++&U2h;G_L5A@_P?RRpYe(0<h|wOd$grD3Qo*gbh%Vr?%|Yl>k4yyPcycSDwl'
    'r~sG<l#dHSzWrAfBzI=XaJ6qrp+OaNA_b9#b~G(JRhfJdzZK}91a@gKtC~VTF6x^wW))}OQdwCfC2C?}{Q6+!!AH?iArm>uQFUtx'
    'VrHY{;!~U`XlK}v)kUK~K6TEVK#erDWAv>Yn&g6QgU&{FwcY)5*a@^}D7|c`ax`Qfc|s`QX%r-Ha4Y6w0zxB*2Y<!rBOtnRVv5v7'
    'O)H6fF9mF*7jrXOR9+hmYWKxS_(PCQ+gg@o5e#^W#nQ{THYZIfCXrSHFFMXQ+()qRiXH!>$L<Xe_xsTcB-|=(JS5>7IMJa1=okI9'
    '!U#H$w=)401pok`ljC>CnJhkM@po&-jh!H+tX59RLU9bE+0fws@_Wi!YTtgVwaf2s8KxttAWKzI%JbNnf5uS@++4T-`b2^>R2c{?'
    'HL_%>Ame%q&?gI&KGFdQt0*)SS`C%Iq=ivN`wl=PxaYZ(d<MD-Co2kZ4)>ka%yqQLdtg2d`jOV|b}2&Vue9H49Sy=K)0$<<$H*D&'
    'P2yxNSvs4A4-ytE*M>!&G`@w&jlu4{Rc3T#EZ1l(7~|dboEC{1TSe((rdS}%?sMw{+_eC`z<QX$;)U|Zc=!WWl4c<xmqDy)9M{E#'
    'EA|sRaUjA2nrhZD1Lwykf~iFc@vPJY@RTSeo^{25;Z-9b6BF`j-)z-{Tc7AmrA@uTa}zK60#Mz#W0Kcoo(a%_=WJ=Hlbi4(7!gix'
    'BHZvr?UA%dYzXh?4dfDX-r`XVF<K>i;I%odS*(4(M4r$7d?o4xbyMn{<9lu<S{~teYfP4ALyB(;h%eWglt@frf=4Ds8QsgWDh+aS'
    'qO`<oT8e-&l_GBxxGXMW;5EVs9F0Vs8ICqB&BJvXt<zT)`uu0UfETH##-@{C84guwTdU8mhskQ%E9mi!7RvqD;b(fxyskJVH6s5j'
    'p2ujl=zH>ctSl^K!%uDa?{a<l#Fn8aV&Z|Qn#>)a^#Ij1wBJDp>(OS9SxHk+2hW`d=xyk_g~z~bANpkIQo|wC2@)^GY?`6}>g?=n'
    '|Co-0{OW9Q(&?P>pC<>W-9!3`M?+ft?fw~^Zvi*H=(W3h`Lu{5&s;f4SU*Y#MV`NE4TIGTfk@dygCN|+dNsqK2aOnjEr<#!+YaL='
    '^O!;3Z^Kk<xGPoxIB204=jL__++<N~GR)^4lwjwnkmIP?cAu>%SnA4xNAe}$r1*e@u_jXYj;3WI+|kg8XbJa9B$I57S263Tkr{45'
    'Zp?%73-r`ugs=HZgD+uh1W-Njd~g}u4GeJnCPh~pvY=BiHrj@@XHMJ`8=w3IOan{SGny87O6~bBYzn>X)p!j=9w)Gg4pP`Z0H`GW'
    '+p;i?;X99DAU4ENa7cX_d$_kg%OOe6Xd&zDA83F8sAn9a>+Vw`<VzT?dM#_0yZ{V!o_cpI>qTLWBL3ik2ht7hqm^wHXr<8Kga<5p'
    'w#<k)Xt=fwD93|>9*E!0Um_w}Ck$S;5Av4@K;5+ux&*zcE1``bRuQJ$tmOZNVav{no?oU>UOSXXH0r{fxoMse`><msz+YlWxoS0)'
    'ym^MH-)sIqGHdYF5?9pm;}F686N2M7yth1D+%~7?QE4fL7MPc|rY6sJ88!gB@SZ}iq-6_`I-uJ7#4w*XPYH!`m1)2z5?<&hbARnJ'
    'H``t}r5spxdJp-nfSDBCzc#YRnYrx9XUvo^o-m)5g(Ch+Ns|fqJFOv9&Eo0+A(t4)p=(kH)Lx0z1`y$idtR3|;d29!Qze-sS?vB_'
    '6h%bgLKf2ESD}ycK|W~3k;h;U{|CQI$6V!fr~)*8_Blr$;6_==Tg*EU^|KkKq?7-jTa>xwC(JA5IYrKp*H0w4LxQX2jovRi|ERSy'
    '>g;4qidZ1Izlen+c9-P{AZAGcQlt|6qMRaG@w_3D6@XQZX?KLM3>@jyiPK(-hg_9soeMvgbx;$oD&3*ruuaGSt2JK=Qgv(DOpv;0'
    '^@~WtJ@zG0O7Tj*8Dwk@MDkJWX|+&3b6rzh9JsD2<R`8%#*k0yqt>YN-UGvk=eLg-AhVhK=Zto;eKg90v{kh0WYJDl>R*1qlf%O='
    '8tyiSx{?SrC~2bKW(II6s?v!Jxm^h~ms)4&*m2=G5Gf(U0cd)J*XXSN1McQfI-7FqU^!vzeZegx|0+y8wBjR;Jg2t<-P!N*RhT!g'
    'mc#ILiZnJqV+o8bJdJEr$yNyoLBu$KORsvI06ua_Z=IfwIZ3dQJNP*ZQ+`@HI~&}qy_Uwy?<t*#jp#7+-1j0o%n-5A6PaNjm?}D|'
    '+zTMNy!mHD-VT6HN^ka*`)#xty@dB2MD1>`Yf7!r)Qt>|=c7bMLiw@EC@DJzKr$Zyo>rk@Xz<aPa=#E`ql!PvxnOn=ICv}EqbsFJ'
    'DT$5G$KeCik}f+MMRVbO_>XYEDCDaU7$+x~e+lta7epur4=Bh>A<_YK*oH|B!C?7vrIB$idI=rl2c0v&gle|qd&>X;CI*-;MwTu`'
    '1U!4&YZvE>-PpM&2k2MSnq&#d*Yi}hE6}4&NVnEzK!^5&@2xmYR0<w972pp$Tr3Q=fKI!MV<|fhI2825r7qp$-oiZO9wy5T<MGg='
    '<j5nsZ1=F_RW*)+mb-5f->Mgj%(gEg8)jV&DZqF6NsRjZo4f(~qfF{a+?fLcdw3z~gnqZX^0H9MB$Npdl2T~atO<!oT|82fh+n*@'
    'rW#nxRyoD0FRP}uAtlnN7f$J_a6Hh>Zdby`MZj%Z2koxf3>Sn&cD2{zR}}eDPpP>>cOds0nj|Hk%H?~HQogbhoblQh<sd3Hl?}fG'
    'ODpi5AL+tnx80JA8xf9Cr+z%XL9%tbb(2*x*=bF+Z+HdJl94lM6#Oz&1LF9AG;qr@ySjlcAzdZ<W$om}p4RoQ)FG~h)`8S9-tpGK'
    'D;-oPzdHM=69@(g!v!rB3mx!*&qrIDm*r?BjIz^4G9B`6>98m1RIi~V9x@n(ojP`6oT2foJQ9Y;LlF{CFd98YpZ4KigET=;-GeOP'
    'U^dJVcpq<a%~#fS_A0_79Y%ntBmXFXakugU9xU%x&O`5O)5(h1VzQ-20Mte<4p0E<a_^LuBG}JAQzx6$xn{e||8~tywfW5|Sj*6b'
    '-?l|pg1c5zuWdoKjnc4q(xwm<<79AXmfJcH(p(hlzy5^#*@@O_{=?JG?(?t36mn2)wLO5^#mQZUP&pY`JfLm!4rE4-hL9Ow1^MD('
    'alOvF2;uG?RDmh_%;BM_Vu|AlmU!v7vRL6(poVg8UIGj`@TXOAc65^#UU<PhQ^j)Aksf({{mP-;x|4my@>IF^E9YIN)(SWq=4&Su'
    'zy1tn5gZjgG^8F+Xabw&;SSd9o#WkZDc~tzmyqD$*YE)-BC*LY*uTIAi^Q^aXGd4r7I~p;3o}<<eMX2MW|d}-vbZqziu{v;y0m@B'
    '*-(IjCL2?%(v*<6r~<85ReL>$+j&iWR9}5{j%DYVG>MfVXk^SmJ3X8^g!;2<m&{%AyO=$xSLL~GH`jZ|+>$4k{}*A&XV&^Y(%jtw'
    'jtl)sygIVli)XzScV%|}%GCQZXE9QUBU)Wz%-_7km-|75E#7FAdDfucN&uAY4N<Qm9Ud}j9a@Mu@N`ga3S35$#O~~x1r%<zvK;}5'
    '6kd^i?*uNtIy1pppB&S|Ra^Viz|5*aN1Ec#Fl2h7mpoToo#myJVc5%_@|(cDjb4;hEvzOZNRw%=^m&WY4###UzLkDvkceLlHg0*h'
    'd-d9Y0RfSEZ9HE!@9F$3mw@khkj+j*8&>eQJbBk_A=WrNKukR&>qf+dGjbEH0V$%1(F^o|IY0RdYeFwr;;XLcbMdO(HES~ginwMm'
    '$yG)(x3`VM`Ze={9I*2t;$M4dY|rVZObvn+_?F-_Ei|JHW>pblAY0oEBmfzX8ebeq0lHe&esyd+uMDJ=m5_mCKB$^AkHaJ>QWHK0'
    'H^!*-U;p&`QOBhG1R$~2p2$$_?|(&xLdvCZG<!Wb+2PeiVnoY7M<PMmR?}2-G`xA9edluOY69kggK}l+t*s@%_yof2M!$dl6g-u_'
    'v{Nd4?b9YxdF6^MM?9aZKnq;3@)V>Zn%U4cfsLFR^@nWT*t|cxsU6upw;P1$d*+60D!%CH4`OIb>97ax6k^~ob5NUIqa=OD+H2S5'
    '*l=2DXziBOtrUwzmZ(8%t#(t?Zi+T?qsP4I%K2Rw+nBcIxB?>uQe&}oMlY|^_$)8g|JrIT@^x4_@BGrzWwvqY(%$XXE?ru!{cPz%'
    'XPa@U!VkyN(P27HEAqq*4d(Ik3p%h{e#Ie%{7yQIDJP-Jf2QT{MHI}{FN*PRgwcKvO!5-9qGIW5Vi11|DWOG9*xW?fkcsZ(oi;Zx'
    '$kWJpmiUZ5AY$<Ei15W9f0Z30x&}9>Kh!V2Jt96OVieV_doR7Uv-_M@a<|08lm);jpPk>9Oyn+p7BT@z<3?Uj`Y79<PdD`zi;VCd'
    'Zs%19g8O^E{rTRBmr3tQ$6JmuOA_&gD{OE3U+#73D<(0671KDpZ;4To?i#D<*V<%$4N5inRu<lth1sgsoa$_5V<Y}$Jhc&jON};v'
    'u23#kzgXDmG^?#EU*Y@85B*8wUBmn5{KlVTIU-B8$o-7(_-?JlT9X$C-f#y9=nE4i(20pdH7-KPY-ootgB*6?M`yQbN(7F#BGBIB'
    '6k2;7__f>J6(ppcC@#)ccA4FpDN;{dlt%g5e)b`SX`LOHqA%}2aTx*@SlM)6`(3mwe;RvHDu1qoK22uZxT|j4rUmX9`geFWns1|v'
    'iLCPZkDf|O)W;19wIs;)D*p6`fBEl!{ZIa}5V2{Y)U7?8S!IUjew>--veu5I>zy3L?Zn~zPD}su&UzM~NuWcTuYKRB_Dc_&Ke~|}'
    '46WR*v&;AC!~xOTk1K349r*(f>IXum#r39T`Hs8z{$&RZU+m)Tw)da6y<#Um)k8%2CxQ~`f*l@dkO%bTgIv6RA_mad#=l6}+ul@Z'
    's1li_(_sa#8miTE$(ujCv58L}Cx4OoPPn=XbsBX_-@=RXBTK0@_roM3m)~5$(_9Pv8rZpbh*)c7-AltG?MdPJOrMfoXf_>XvTceB'
    'wvF-2w)Lazvkn-|-oE)Cc@F1P'
)

ROW_2_SOURCE_B85 = (
    'c-pO7{cht(a{u3e-+{p$LUKpuQj+BlAv!>o<QZjT*^oT5yQfhIN^D8Y@ax!=ZH?il2oMBFkY~vM2jp#X_Zq3{=122Gwq|!{0Lf%m'
    'S5;S6RsXuXv<iJ!GK|$W+J?k1B-;)AFp^Bq^CL5|eUBa=t&k=#qm5%P5`sH;JU)sa=w=()PWtGYQQ-IyQkztA189K{ZA*zFch_rK'
    't)SFaz84v;>6vR1VmExg;g411!4CgI!oZ1Q6@})d@Ax52lpWa~F~ccYyTpr#)x$>D3~JCobI)V^cI<DY%+X)1D2W#SQ^K^Hn&=AN'
    ')}&iTFmyrVRAF!H20K*J3(|Li=~?gwHMe3hcf(P_mG!UjuinNFXl7($F8t8Ljx?Q>MQGX{4a_CcEHg4efppN0Z(7C@tVqr-Yy2rN'
    '!zFPXrgWJBjx@5pb;1CjjSX>a8f-{NK;)<K@On7t8`rnvdt*BM^FZoI^$SfuKDz6U2P0$Dy@F4%wb9V@qwdecsWBRk2gXms{{08+'
    'q&1F@?uWgP#n<)Yqki|kTYP<Sd^DI$ZYLF7u5N#>U~)C-_C5~oE9kF>?>|lAcgEfD=fP;oMxpCT(}%(E{fB#@O>gN(z3%vPcWQjN'
    'oecjB<Li!$o9^WOaGa5yH)1RFj7grsMLlWi8>7Ls*m7~2wiG&5>3Jf(p~lg;dyl>Px1X*?1EY64`gAjndEDHvaW-+fVzTSuxSOA^'
    'b8MY-r}u*iXK74sKaKmt@%va1(CC_8FW7N7?tK`b*00O|QEIfbQ|Sk(>_|UK)jEM|O$MKbgP)B4-4FyQRd#tNy{57X%qz4#F3fj2'
    '2Bp#u-j*RKiZ&egZ*Pq0V9>`f&>Ls^dE@f<=;PoQ<8E?$*L~l;AKs1)V0p)c)@xF;CbeqPc}=>63_0jdM!y)nPm|Asl&Fr8CUge-'
    'LQLFrf9A8Iw~mkAgJO5a?W8}LAXoWAPp(OFciP9S%NtoQWcQ4%ZAtz^<Kq0{s+ADpo59)XRkJ0(93M@80qHj)$GSm@NBwYz9X>P+'
    'Gm0RsLP}xJO_p;)-(gRuM*n7vqGL9q?2J!dha5|6x^A8KbiKfx_33r>Mo$ogFZ<_By>Tw6&azUK%+EHYVb7AL(6d5rch9(>saF~z'
    'yu>d->fm#=)KR@k*r-vKGHT`pj7|mq<H7w;x08<*5}t8j*mLT;H-03)l*K*EeOMl0fNX42)A<Zd?D3_j#WNc$f1hXq-O0+_I`BGE'
    'h`TBe`KobG2R-8z-$^L;`%^+u<zY!1e-e}$jPED6!@j5&-NB;oj!l=erIjo^z8LTaEPbY9ED1`mEg5h~C}kv)?MY%pWRj+%+q=Pm'
    'VgQUdJA*oYK#aOk*AI|#d}NW81Os2$Yd~r@DKJq~>EurUTs+b~K9b;{8QP|U@C=l|FGcRPzkK=0_ALLAe))0@wu3K5rA5&SgEQ@R'
    '<vZ44T^ClV^DqMFv^=c4E%pw{U$%BgEF{2w6!T0vj9Ol`T2WZc1YimU@FMb<A!cQ!Wq*KC%aZJs1Tc;fXyggtAXU^k=Lk(jrm=~c'
    '(xy#`bZt9i>_^xB*0Tm-=!c3tiDgJ#)RW;@X$53ddIS{3q^*!#ONNgMu3`z5N(xd1RIG`Ao0f%-3XMV~;iV>)0qh-usY27G+BN{3'
    'NR($8skD{p+763OrYr^&drwV|o=_6nD`=a=jdN%PK)i{_Q^bS=+QhEQEscCv7F$C%$1{CzN9p^v&1NvyyjW$;3z+XrAAsc0RDuoA'
    'ZaxvJMRo)ReuKy{7sPpm=qa+7U*DwK9Yp1jHGwH+R@zU+WBE<(W3m;1bhfWSBRy{Hh{(y%s+*|!K%nny9Ua&iJ=#{Zfu;gXPe^D0'
    'G+<&s0Geylp`_3f+UreJY}<PF(zl4=+TNBz<9)a#f?dED2}}A(C;G5v){1rknnCs;p`n(6wG!W-HeS>$8E(PRLq9OrY)8`#Lvu%!'
    '<u-gEv=0%qj9{T@vsA>Ar0qrRB;&w#*OTbhRg^jlKm#MM&{bPI{z8#|Fg_cm<7i&6lT|ep@AylIq@fpVd6X&;iq4MOm7*<s8O<O#'
    'YwXoLRwD@ge+ANs4>_SDmHKP2UxRgs&Khw>G9$;NJ%B7W1~H~MSW007t0<*RC$QE+K91>fW0DmK8+!f1IMww#wi~3z`ItnHe)yFd'
    'erQ21iJl@^#q1|{uZI0WT}FPGSOS(*)HFL{Dl4dTNP<lCH~}4`Ux<HlF0pWuB6Hyo=@GZfr6gIHnw$}4hm4*xFai=9bi2UT+#rp='
    '{rms?{jdM_xBvdH!gv`%+~aDT;7^i1&)H<Ju7_BMt0RG)C$XM5nnZBSn<>S-5>`W0W#8ITPz8+Tnor7J1#8k*vg=fjHeIYK0KZd_'
    'Wyx`2GgLC`sf=&mhyFI0MrK3-tFhbwkdVPSlSYQ;U-#i-iWw>wsL!%XqR?JqCOJqNGIbYb#DoGcL~t0hK&vG#Kw~m(Ngvo^6WYXb'
    'P47@KO~#*_q-m_|5C#DgVm|<cOn3pPq^W>ps34$Z$V^kgO0Xd_9>7=(+cJY0g2jq#`(|YRPOJ?)PcpWYScwq24FZSY5t_*m{r5^i'
    'B;}BF8glFs87!TQ*y4(FkWVf%9U(};d4a6(CPm93F)fy#?=i7Z#ZGf|yO@v*Njk7JH03oeQ%SOB+36$MF?5>xqv$l^{1zvrWWT~-'
    '^C&h&u$js;h|D=la_|Rcs7#+Y?G$MD!`XwO?5#&65a{}lTM|^71_oDPZ|1Qx!{crUG-{(+aZU`FY^9*Bx_4;!?jJgNYrT|Y%yR8>'
    '5V=QKo~|ZQ5_IG*U+6oSkqyw3u!EZ^HJ5(`LD}+eTFlLbM2R!4#Gb*Z+c{6FN&B%$v<2UoWgh4SnrE@x4m!U18j#Ig02R5!o%qr-'
    '+v9kk2iV2+L67ZfWj`sh2yfa_X)Qef6uk(G4IB5{GUG(SnX~Bs9@!PWV;x{GvD^^rMwkR#^n^zZfZI%bF@wll7_ir$Yc_<3QfWX{'
    '`bX)tPKB|nb8YDdJ0*4_@)W=b2zQtsF0Ztx)f$@?JB2TivdZ>6iVC>fz<EwM#p5e5Di8WlewY=?EFs=1R4%2A&!)2_ajv-A16X2w'
    'V)+~zfe(aIv?1^p>`O^R5-aeaPtymDNHB!NzhvnH+}@;&lTvwr3bBA2gbu_3M3YqKJlNgFbn7xtWcJ{vF~IE`34N+44)LlPsx=AF'
    'shaRzL9|CaMWuYuKZVm)F$P|hSvuIQNvMW+-lP5s!Rwqyh@!zLN_mcFDaHKof%SZ6>w0Wusyg+XWk*=^H;%OC+1bZF!GgOnk~Ekr'
    'yq(2#`TZ>RepnAWnVh+Jo&rj8uN_(_Ka-S2xS&W?g5~~+>p^wun56<BXS=Q$7%uFLJ93!#dh!oqvB*H98#5pPe58M3$i3QG3kp^T'
    'Xtx}e(gyZgk?%_@^2j2NYKQAopJSSUDEZaWEuYXhD*z?}<Ku>q-%*woNrPE%T<KjdH53({2}z_TwdeVu%Cpb%cL6#S61y;&WlLck'
    'XY?w}St-~LG**&H8Mk<2{Ptv}$){kWOeYp7?eg6c<jh(w#OJtBu<L6@G#8r%7E?FG^op6La*lo!Q<K<mXwum$u5!4)MO=*bl)@-$'
    'MLlYwjPyXrw5&VRBM!rsiV5hujl~IC_X*vP9dWcJ2G=%zXn#YWckX;@qrbK#X64R_uIcBk>`0#<`v5Bq4@i}A#~R*u>oNP91RD;I'
    '!bsf??}U$B{>JRc4q4&O<N^L%_z^#}O^zr#l~?$@{gL1iA9iAAvP;Jf4BzXpvpY2&J&kq_>4MP0$V=A~OfP%+6{`_Umn!V|u~R4p'
    'U~A$U-k-u|%(OZ!Rp`AAZA;<86Wd#%`N)jlD;fSZJf3XbDjExO0EV=D_dW4QXaW@FIiJ7OEPfeggx*>a?tK#^!`8qaR-S|fETf!j'
    'bu7+kX{Tq6n$*<JTdkVZ(Dc)b<d}^)+Gi4i#Y~5nn6NO2Z)$a|-ojfjyvIGkhZkY9%0v0iej=&6h<(pm;A~0f^TwH`zhr$U{$r`5'
    'a&dtj>P`N|pBnh2ri;14hYMMkJk^pXKKPmd2UHa!kU)Y#dlKcF9pq?X-tvH`onNzafvm6r@sSkQLT*3It@Q1EKowX>6)9dNieP6?'
    'woLv##Pi3;hR3$TP^EX3lRT?}D<ZRhwj<80;F_ApikK9b8<<~oC>Gv=A=p-+pdt$_y2QKJP!VWrMdjZW6ae;}3&;t0zt9m241@vH'
    'B-05X0qI&3)Jg9gI-nvRkPt{}#OmM`K|w1QvMtjChRF=8K=j~CQAM@TiAGpq74kz2u_KMF{Ln1RUv=tOyMe8L2_vOsr7ANKWV|p#'
    'B~`CzrJD2vM4g-_kZoMZp}w3LOI|qU@+&-&!n13JUm*l=)>;}EjARxoRzU$TGhmcWq=biP1G%+=?i)MY@0?Z*D7lBiC1!8t$S<-L'
    'WjMGL>3}4v^T$j`3<85(kxdOR%2T7ZwZ`g2e$%yZVOuh*$4guMAg&mASFR`x9c_f4u)k93EOnl<hj;@1&}-p;Ip?F=MQXe_&wYo@'
    'SY~>|UM)!Ydkqq&wh9|?))nK=E$z9b`}@_0jQX3jXXQs#$|K`nt2?AegWZ`b@_k&eAtePHcp?jlJ;gc|@CETiy;b1%!~j$ULs-KA'
    'YpXeTdSaif2QY&a17Uic=f|#zX?{&pt7!O5vNFq+4Sml>TBssWkl~0UL1C@MF1WyQ@Kg}27FJqe{Y<$rZP_d;xjHUU&Envm9NSgu'
    'ms$fut^r>KgYU4N0g!X;vQ_HM&4&z{;AxlIB@bNqtZNPSnL?yI@=Q3Dz&M`B%Q?WvPeTyeGtv@r6t)@+wy#rqzCyHx<5)2G#tez|'
    '%gA=^2%7kTE?awQd<mKtp)fkuc<U}QTnJ0~nqVnld588+s*5w7gIR^S&S2HPnaz@-phJ0pvv$HePTN07bzNtZ#$$S~H;5UuP|(Yp'
    '*ravj6FE(;4H%*zO!NqrrP%E2+U03Y1ZS^rcyhS~D;VBREJ-4eOVPTofqQzzoRv8M;SV`K+3WYYYS4X-&RsdcoQ-3?!V4IxVCG|a'
    'Jl<4-k^CV4f|YFKU-8T<z7y^xL^sZV&GQu%nef+F4ME!8#Y4%ogdQnlbsuL%WO|YgTHPZcwI7zrT(jj+;j1Fg0sK`>^0!g2ji%t2'
    '!gY-h#qmlS7uzTqx%OyB8zZ^#(CEWjhGd%-<8<Ou9QFIy1lH<zVdF7vtibLFb<4k<Z)C4l3F9+euV4rZu<aOMAI)%0JI^xS1B8Uq'
    'xCRva8_u^!S*Y^~3FI;Wl#O_ngxk2%(TvKs>v&NFi+HtTRn>j-QSSK#z-dFTX#RZ*#xo=10VMd}Ds@SQ@0E%Y_UfU$;vE4yBEdZo'
    'pH5gM4J_AEiaDfShs1LR`KRh94Un63T~#xV{T@HSW!urWSXcZK0@wCLRKF<dqE#(Q3-G+3WlgrbL@*%^Q=nutkTMN_wNi34U1}Ym'
    '@@1pRtx<2{o}oDU56eR=+HC2fN&6KhjX}h&lPbr>2&Y?o25b4N@1z)EUU;8YCZy=hafIcagN~)|VIiA`ctI&dQ@qL0LI`^Yp<<~W'
    'aeUIOswqGaj&E-u&;gXNUnpDvfs$83!)pz^v@5>0JtbZE#3d60j&JF!dH^Fqsv56S)iU4ZNMl)5z%ZSTY(G|MrveC6-(dJ*neLFP'
    '%v#AURfU5hJTGoCSOlN)AHRHwY!@cBuK&yPN%Q58vIvlbu6dNi3pjI|=h7js=<T`VTa2rYjP)$ILU-bNPKK4T7lr=3aCFA^Xmb{x'
    '4>q;IY13ofsxXdWAUoK)yu@=)t73ox{od2>3<`eN(YN}lS}l%G*-Q+u!asNAW$irE<rPD}P2)o~Lf1G$jnGg+kqX`i6AyOyyYEq@'
    '>VySwWc&-r&ZYjUZBEZGP?LHce{BLgj$bVq&7g{alc(JTD!3-*fEbun{#YVTD9g|eBC&>%{a(Q^uu$JHR1FY`?>u0=1h2u3(`TtB'
    'uv26e)8lPmu?5Ru@Mu~7`(OX#|NirTc?F>?!6uv(X}LHKvR)B}6?f*swn+U%aJ7?-4ZFW^x!IkB{^LnQuH>e<Uo9JJTv>^6hu3Gb'
    'k}<>oDxSq=wNs$L+eOcn<zM?N@nUktc;k|d$=*|O_3By}SA7+$2IpD*%Q|2968|dhv~iaH9<JFqe|fd|<&&3`%SdtkE@R5;hO>Lx'
    '6yJRxq>X*HxJ{{*9ON)}NU|$n%%$VggkADF#CnmHJOyzU1o)%WNc4{Uw)9+(`NDq9y^;(A-P<sjgBu10qG8DGq-Y7l^7!cg0S5g|'
    'y#'
)

ROW_3_SOURCE_B85 = (
    'c-qA~?QSDEa{mqZ9XiMX-I|sATJq-@(E+j~&j=&ShUA&uiN_&mH6?MUzpmM`t(_RScR1h>AVHoX{}0H+1jyZMq>AhhcB^H3Hw)}y'
    'U^U5NRk2u~BzGQ%zCp-*m8@b$h{63Rj1z+fL732lhk@ALnIlX@lcmS+GJq>+?(U=wv0NqG%bPw;A}>r3nhL{@U<CfyE5R(YyI7cw'
    '8c2H{1_|+LKo=~=Y1qC{%{uUCjb8{Dc}dEmHarWxFculJBOWjsPuap}LBiY~4*E1|!hk5X$N1~<pV%hGa6T6-xeFgNq}^OZXVBX;'
    'E+xRw2aa=wgOwkxk;x#+`yv{+(1V=2DVeL`D8nk}H`PyX8AhTIGNX549N<Ke@5)_Fc_1P>W422Z3Jm0n_Co5C8Ay>Gy=|&iMB^Fr'
    'JjwJX0$gdrgGGh_+hoanE}|ui8Ib%q9-a>eeR6&|z9!S*KM#zK(K@l6-JPrMcrYTP?iu{mN84>2?fx*FlF@KHAm0!B*Y9zb-QL}~'
    '9`-&|yIZ?E{qA+Q+I_OSGnh;+CpB2kE`O*&ayIGqJ`Aom;CH6OcesjPcibD04C?Ii`ug&MoI~?^csT}8$38west3NlyedPUv|44z'
    'R1PJW!=s(gq!2^}$%}+`&WG<lPST~2tKkoW(Ns#`ak8QJgW<dP*V>qK2(lk{uW_({`SENtAic}c$BS`_<>EqO*tZL?^WnH#-oaxW'
    'oph(ygNZ^-CYK+_{o(i>x#&*b4b}1-5P6Bi;kfsHfLuIJ|My1w&^|EU7@Ml~d!wEwP`1h7({S)T>0b?jfK?}%?~LbIZUFKcZg(q^'
    '%-3MmO88|O8lxz~aUXPSIvDiPFr4<0bKHKryYpf2nOsdSue$HLxX}bqUP+>@rm^2N4x7et(|8L$bI_fPK9k<Z$)`aM)I!_SCWCyz'
    'NG`fRs9kXmcX!?aV^`#I(jQC^tNFfXHVv~o?V}%7gK`$4`-Y>ZhWU@}ljD=K!we9621f^H`-kS!?#}cxP`}VIwuoSX2;((Qc;6;8'
    'Nx;#9lacLx>4oCZldS{NzZfIw6pc>1z*FX-gc66&5086}Q^79h^tw*FrxC)={_(!kKGs;5_E|^s69>l}xW*J_*3j*4Sr-iTHkJsT'
    '_@hxBe5$89YSj@NwKu7Z_R9oD2O9qI;QIT^$%h&W-*90#a~k>=VPZaQlKW=%Az;D+<=Upy`vgO>`BdfNn_R5w&p3hUY))4mbQcUM'
    'RaGWmx9;hnN1oB03}U-JWe{~5mW`>F#?)YZJ-HnAb-u(J(g1%<eRgWhO|AJv-~)mo^~j7N1@4-VnZ!l`V(`GwEux?_9bH}xUP%Vh'
    '6K_p~7e<V<NXvNzmE9ee%?()ioG&1EyU0@&t6}W@0aBTOo$l@!@E?si^)O)rCh*5Hd#7K%e9r?nd=Ov0oP+G(M?%?2GDqXgr=5kK'
    '`?{%j?#AR{0?E|ub<?>%J7#}e@tC;?fb#_UnS2?|GHcB$u@ng;E&|dS9mfr3u~M}11BALsiO&s4>?DBA9w9*#R^wJ-grO?ZRK${L'
    '$^|pdxyQy~avrV%cM!*6Y?+gkhS5bnWoB#40ZbSVkXK1+a|E}u<s*ahltL|&8dWV%w)Jo7y4bWt5?dLrMkX1^#$ymwOnqUmBFG$>'
    '_02>mdrp1sN$Zr%qCv@iK?AWTSj^`zc9Ra?!YHK3%Y;29k~rj_ICZuXN$8vUXzZ78=KVa$`%k&t!P4;EycoWN{Vw={Bprq}(gql('
    'k4)GJPe8yg0D1I|dCvenCVckwMX2?wpggu<u*FhJ+o6;#zsP+GHixubRyru8hb2##nJuk;NYHx*bDx{&L9O(_-DC+vmApLz5fQ+E'
    'h{FhA&Z#F@WhCZ{Wl|km1$-8|jQBiQ2^hSNS4@)&=pvI&Kbs`(7gAcuIzlm+0XQ_|GIHm7|3UkyQ8>dDD0&=5bRlb;ZXDCKux6|H'
    'o{2tK&@6#Kb1Getbdo$sPBV`K1#n=9m2X)sX@CSRuQJtMc;TI8z9F9o^*lR>)@GxTllQ_ISkf?vR?12(Fhy@Icr9s{&ypK3&Zg|T'
    'P1%Uz@J9fhgy0i8#>RXbDmaKk;-;N?B${|s^dM#7XvFB^Ae2H$YgrqfPD5?SY8}(rlCn9A+fM6*95~LEtPce_9<$^jjK2yJ#xD4h'
    '<S{WD=>5#@*|0xonTQV^OC+62BlnKz$|@|q#zAI$>;VtPXBO@q8`7MNgx-0~c)$XB!%0@QrX+-$*Q{P4AQ6j+Sl!`hX_4e_fBCP!'
    '|Mky*`_KQVtXF1;TU1Q}{K0T;OETHY>uaQ=*fD_5y_8Q}O-4AyEt!&93DFQ)*%n(Wtbnk5`e<!suxWf{>rVY@^I$C?@jK8_ZYVAk'
    'LmOf}FtPVt9Im2iLK6YGn)D5j63TRrii8B=c_02v(L<FC^_%IlB<3^pB(I`|Nc}sSNTQG!Vse<GK&fRaAW|@UChldhi8*t98oZXw'
    'zQnIYvQOqbhDE@J@OwxhGhC2Vikv{UR3NY+$bwT%N>GrIdk~h)ZKc47Cb25oK27Magqq+96Ilu7W<<m)iadsgYE)+QKbHz-QXb2<'
    'p@c4@A>GMH23OpJvU9m{Fa@b7FMw5g@@zR~)Rq4E8Xb$Uc<!rDs}8y1q$AfxQJ!O%%AB?IP9NA>W~aG6>Pn;UZ*@~P<X2g2Ig8CR'
    '*n;I7u*_TO<lqB6RKZW`b}F#j)7e+2vX=offS@0OZ^^RK+%OaaTRjgi1dr7aaCC}d#XTXg*;a+y#@3}_-#>K9(t6sUaZ?(<1(I8c'
    'mEoF(U{S~X@<n_HJF*0L7I(0iYLxtsMkrG*$hfa)z&we`q9E&2Je;<>kLX5aKsz<0GA+ZH{{*@K<xD6zI#ehKy^toIJ;!#A4>t-p'
    'z-41TY;DeL8ZFy7Nbxrf(62y%;QWr7giXc3l6D~3nxi7#!{=R?s2n}ZNISJ>xN7`?VMd3OWC}j>ctpaWBU80J{SnF9V_l$HS$nY>'
    'Z84BC<8d?;AePKFJC(y=SP%;q&<{(CsM|Sg$jR#q9&mr<H{lBq!52mw#JFGyS$o{KQb1$`&;1LqY_9R0vGpYOjbY`8_wpHyZ09K;'
    'bU$AM^U@Y^6`!cLVp3I$7AS1N6Ly2mq2lHo{8LARhDMd!+d@*Jz)@zPp~1+s`$lc=t(<bt1qbD{Fw1@(@(ztuSej%j%xKdWF?TJs'
    '>)EFBMZo8LMuQnsiXX{Ur3x+g9(4XR_c&3yXn8!@(tVCHEc4BqH)bt#J#anjVI6EagD?YI*rsN@Q*9|{F`a#7$)**9+$uKPG-SE6'
    '(OVdN#Q<Pi0++ehnWbV11_gO;xx!8%NZv)DV8cLW6-JFQq-m6mT-zx;3$iSo-7@nPt}D$()39E1QPO<*;^LQsRA>QDvSbA><iX~E'
    '8_G-7&gN~8?g^^`uR5@bk1a_JoJCnOcx6#xxP0AH<BB&KKxd36iY!juao*6~Z6L2BO;m1G4|O`J+8Pk6n4$b}W{Pf1A12@+W?r~b'
    'k!VAGQ7dH7IxQtM$O4#7FZ5P^kj5x^1d)EPM_TY&PiNF)dQZvXn~*e#$fF7@?~L-uC?^^JDh)yf*0On|(<Pf2v@@1%D6el$K`~Xy'
    'V3wF}+}g7+T3cz5)^XgJX-je`6M@pOhUM0V<ZK-=EA?G(N^4Q5MIh{Z9)j*k6T_UiqvLBDkDZ*ub*D@mHh#s5VOd>wv<n)PlgE;T'
    'b@YoFv|KzS>7+*^JP6X91T=Y-bR69j9Q(Lw<ew+#B4jF{9W5&M&q8;dM=G!^4{~L^@7FwM0VpE{^`a%U9J_VUC}aVf$)n#)6f$wy'
    'w?P7uIau;OEdNIi^I~YUj!L2dF9#dUAcbOXN1Z>FrGpE)3m<>ca%xtPZ38$Au?!#P9%r%0SVuZgQH#zuv;H`axr?GV^LeIzK3Bf_'
    'egc9^F21rj!}~L5&uH`uXZp<LG<e<QmVNM4m3yA)JgQ8`f~{&b8Z~3;nANgOL~EuX3)~dUS~4#!in~mZ{$YNN2^Mkql-q%vmffys'
    '8+V!;KoJx+N!;Yqp=N-*z11ubK9`59Af_vI3DZVoz+=|!5eo$)U9~0uqK-oPb5g-mlf%oTT3H7^|9~Q}1BJw4RpwIOAni*3TJ@{I'
    'Pz|6C+T*hkQi~xkl`JJ$<V8!3IHVuX66}tOJJxf7!1j4yWrvq)c3vyIW;T5+&z{g=ff1{OYOlmso~vpTApLiApR1!n8uU(BI?9Hu'
    '_%g%Up0+j(G@|C#3_auB3!z>gai1qS@2L>?WF`$!u;8h@02MW5R-zvmZCqw*%PUdC4Uo!G^i~~$Ea{5jW?8!50f+l1B@T52>l}?w'
    'e+*+H59q_#lIb=$6BgTmX*^21gF7CG&Ot**p%<#y(J>!W<H@$5T=o`c;YG${aCLsIRrKlJRn80aG$}8uBmT%dw1+cQsLAXU`|{ih'
    'JVd!eWvrzxWjT7EOuLd*1gf3F)co9*uSW{v)fY7CO^j4c&>M|f0a$uH19@D*CCtbXH@0E6bfkOvq8nQX6vq<uo}J74GD}^Xsf1W{'
    'LD*g;(JGlH%-1eww4F+?+R~dds{pqjc(Np7HJ|fG%e2M)!UWUILwP3+<c*()CUWP$3YtK2<3_xrJwp2b<4yQ{o*_PRoEn6n!`vfZ'
    'A85Q#xqu(VOo;SB2hJznPY<ScX%{n^uu1r^pYD=&wAN6e81HMSw?&WY-fYQPTb{)ow4IvaU-PY(CQ(_yB=$VlKbf#vHIL8g7gDPV'
    '>ZTAq1wO>VBcXVd+hxZ%a%uo3g|T&v*6~3d48}dB2jH3rUfNn!=d*<>*uB%-O0j7iw46pA+jc~g*|JnzIJIyR0z4!M=kxT0FV0F`'
    '#Tss%)oMfbe*)?DK7Q=C+jx+1aCD+NkMJYyJaOKt&X$8;X>Y4_=%n5I_?2}Zylti3NBD)^6(&$dcqI6e0N>y$)OSOja2Fl&sYiKL'
    '=}CQdad4Ft!$7KZ)Lpt+xV2*zKsIU3WPC<5lw}JE!R7LR%7;^`ka=oU2&gl(ECH&eVcy=D5xGy(AdY3ZbDI_3n`aOu5y=aijny<?'
    '$RC>c((Ql#^yg74pZg^kAe1p3^b83ppRM7Cq2IXt?#q{i`>-`F=U=|<9X$QcY*yA<KCkG2Iyhw|l+jo{?v~n&+uUk7sgcKkCx=3N'
    'Ks~H%A}kvILCo5MZ*QA1qhGPM+VS>jDlgG5m*j45W3TmFuval~Svb^)6S7$QEf#RN!9}G|oxSy%Lsd^d`;KEfkM&~B*_{6dY?fH6'
    'br0o>Ms@Db1D8Qa@6M7HK5vw+B7c9EX~?yFh*bM#)ks^lmbm$t)yUiR6DsdqjkHw{tUZ7=;A;~oy$)2BPFAKE;I_{D6H>ltF1efc'
    't@rKs`+%Q%o>`ojrP+D$9<zbI`X?S$yj$xO+?)PRvr(<=a$f2liQW_hZi{)dFvfdq=2o2Z=C;ctQjJWrB!J9EZ}HP=i6ynkuC#AU'
    'XRXbudgyG!?&N#1mv30v4Iid%xU{S~Kb$+_wER%PHA@_QiuL<x9>6x=!KMY;Eap+7KTnZQQ3=5}n1nPST8hxSXI3M9UX_-A8_zvD'
    'Ra7ZFR(sVEjK8ul{`+75?SKFMzt!WA^w_=vtH`O8`x#>Rn1bGMm$mk^=RbQ(&SDzRmg~JZeAwGJYq8n>SC`}SCHZ<RJ^GZd0Z8~&'
    'HuXFD?5)Ae@7Zc^_P0N>C>0w3`E2T+ZeF!|Yi>yDb#h7ZVKxq`D>8IhcgzEbKjsw3xdzNI_A@q<@YMKLq4^~LAGWzc2<|f>a;FJF'
    'jU*&H?oj%<yYqjAOC_8'
)

ROW_4_SOURCE_B85 = (
    'c-qZe-EZSaa(~xf!EiTNa);tb^2c}#=m39d#(?cH?3vkIJPtvNEr~n)T9Wd|8h*&j-2ur%kN|l}aM{Pa1jtK(AeTSqcJALuRX54*'
    'CY!S4nGMc&tC5=3)z#HizpDOVPtrKl9A`4m=P7X<%@30}%{4cQ;@r*sILbPm3DP8PJ`4P-lHe46c8c%WY@Yi;r~DbZc@o4q61$le'
    'Ch!D4?0H5EeP=q=O$DVriKE;J-N>Di6szIyWA;-)o!sF!q)URF$tOG=$3dKCrKCrGMBMb8OhXdo#M{G%p_^FR3H}@SKa#S6XdWha'
    '$TUjIqQs3nD8Pn1*3RkSQOWr*1o15P&)zIfGDdNgx_*=;?wHt~o4XmwD_Mfr^_($ujcmTO*ssJ*$0P`-t{e5SN4XzO%li1+nUT=X'
    'k{L+}j7?{B`0Cx+f%Dzr{@d5i>ET};9Gz=Jt-salm5*K@9KL$}mOoqD>UB<b-r}SEH}7_j4xGI=NAHeL3LeMDwEmh64Bni)bw)eK'
    'heyALGJPlY-9YcWJ2@O39_%~x>G|PbA5eO`VmdlJ**TJ)zOZ|;oQEfSuMbcbqocv!LTM<)b8WC;uhXyJY7GGj5_S6>vvam{{`TO^'
    '**`smCRGZ1t|`Q;>93Mmsn+KjZ=}<Cb@t}nsq^M+|KJQeslVRSEzsZjJ{bS`uYvBEKcvFzulr6$qAX7DbgL*CIBuSU&4OW6Woy)W'
    '(m1F})}8(16ZlmzK00{M#`(U%d0W$u`bz%zWT0pT%hzgcVc9yAJ#<c-)19+D*4jE4-0|Vb;qkj;=hS(>bM%hdh210F{s8RMBNGj{'
    'PW&kV&2i<dhN*Rb1+Ey8ZQ2O>Jz`5%sRsYB#7OnM?GNAikr&@&A4WizeQ?nt?L438rnMaLl_!#$^b_1E11B2^O<$C9fIw*_l;%f*'
    'fc``SkgN)9a_eV#W|)_hDLk$7-L6Cg56_4;@&j@b=c9NYc?W44r-pt;ewh0y@igjdl(#kkvP`=He=phta$7eGb+t&x+Y*q}7Qefm'
    'hd+%hPmQvg3D`_E0(hN{Pu<iFGkcx@j1fK<_tiIQ6<F7vxS=1QfeDadDd;jc%DNdz{Rz-s2*j64Y5-@TYxsMO*1aW}o%=bcX@v@b'
    'dqsjJE8OP(_~S}7?pCf5kST$I#f~pogZa`*J%=bc%gJrdnjtwl^LU=+aj3UZrs2Y2t>}DYHqq||(uZw7hX^}v<$X0#)VYcSPgi$!'
    '0zr{Rca-~P=I5kNn7P*kh?k+fo>DF3cL{1vj{t+}G>JD6Y1Rk)`9lF94B3Ks3_fxgC3EI}MmA3UB;N;L9_IkP+Z2^3)_MLozW}GR'
    'XwfBeRPf2*F7vah0$@;$g$<bkpfx3j&_54yk(j$@gH`q9IfS6y!~KK4^XZ@e^G|>Hr%%8C*L7Wd`YMg*$vNbGne%!*^rLwO!GneZ'
    'H_M$U9__=&IqFhCPrvGP6>|!Ez9s=O161_K8K}>Ylw0eA^bBM<*mKU|KS3l2pUyQ%w&x(K2E++To&skln<q&?FuilrJ7=_i<^*vx'
    'wa3{t@0}1I^el)I@}2LIhD>UeXBYj;x_CeT+yDIhFF*bKyWf2J{XgjShlQ3mZIQ|ugkc-yAOHIgpZ@D_|Mc&F$3|t~oV_{x<NyBj'
    '=^y{&)8GI0(@($Mi^E+vZ`4j>zYDo-aMz_>>W<^^%FVl}d(*vv?zL+GD(-_BLgaU8TJUG7g??QN%Km#*Ke~{sceA@JCt)`v;T1`<'
    'nV%??@F2$hD&{9;yibZ~%1zt8c7r*A;9f@kWKqOlIod-<yVPH_0~}5bEMKET%0y_3{waZ3n<!?1Mk#T<oydC+*z1+Blo>T|-L4}M'
    '<dBF@`5c_-BJu2fn7oW21Ar+``>z7OvdYijx>o@KZ2}RB;eh%L-snXG7B0;e2DK6!AoikCY3<>KK4J;TrA>Y?bm|ihRiM*%`n^J3'
    'moPI-?u<frq6;jLS9X<BT4eoLPdUt7mczw{8>XQ1Frsj&?~L=g8vw{+5l#|zN$ygpqxkEvnM2}G`(s}L;Pi*m;?O2)d~Gno`4t3r'
    'qpm{MN@mtg^F1;zZ1SiuL|#|RX-EC6H1Ddx0>dFgC4rZfRtZuCB&s)VYJUNfgheeM+$I1xz?Iy(;~cyIY(3SES{#ElU2jwxE!t!?'
    'M_JS~_@BmsMDt=uxKXnh`(QGTlN*gq)Wrt56<n~;3Fb$~35IhXAu!l^>>yu*igJ(hK+GaGi^<@&wl;Y2ut>0X5h=C8KoX&%(7+U%'
    ')L2tBM5yWBlC9kmBrj)XZZMt)fEDNX<k(FLxTrDOmXV<8NGbx}SpBJhHx!H5SBgmRacOB$Vn?2vrtTg5;BX}r7x+VUQORCF+ah-8'
    'e9Vaws&A|&=P%%9S^`4^l_v8bFe*c0eB1~m!n|-e31T;|0!uY5W0B|tI*Or0ob~eoXEbNAe+JW3XkOB#jCc)I1HjPQ4v%#i!$01#'
    'SyjJmws{zVmIyX0i9sRHftfS8b$VBBMm*pO`5$x@;4P;%O;^ec&Yd|U)P4-NmDnv~Ank%wfofF0=iPd68VD~{P<6Wmh*DDOk^OC1'
    'zjgnc=*I=*%_IMZIjPX71U?vNQLfSrWxKj4Z0Yi%u6r~?YCfve7=*ki;QrF~QH?#`9o`OA0uL#&t|tg>&V5Q-B)8kWHSaOyVV_ER'
    'wA{MP3f*EA>Vc)T3+rF9hACBH>>L-Uji}HRETQ6ST2fapn|zOOXpV^+Ra&i@<3`K8)eeEoFwG_tI%+81l78RDK{?j4n9Zscf?e=_'
    'TLO}p6%*Eoi1{)DOR&EJ<3l$&r7?`veH*_gX^xUH<74sdh^|;{3s7kZPgza*DFty}z{dL;Sd-g^F2y$NZ006pfl{wAeJy3gJWHso'
    '{YvWza&agVi7q4#5{PMCt=|x|MqhC&o=Qz!imDfxlmRu3lOu?ysT-WSerou*Nq~u=uoA#mJu}F3#)NJpK&$j$8wN7s)l5j<B5Vhe'
    'l~=n3%mlmv^e4{5#^yPu7(Gja|BD1LKg?%x(Tw=hS*|^Qt_@6KexJCWhjSYL7WF>VwrDBegxG_oz$5~I^b5v_exmv=n3iU4F!>JK'
    'gCap>CIfA+3Ywg4g+HkqO^GoOl2(DJ(3r?o?mF#$BbwBk`IgIQHHdr7BBRdWnbtQKa}h;knzmYE7XL=qt#;k<NA&H;t4t#zH_QyI'
    'x-RIFsa39caZa8;nE*{xGcvksW%Gx!rfR%pd26!qhk~UZ2@+iUnGe{0>}DUK^_o0(5`$ODtp{#+<+)mdc1K-nTN`X_U>xT~00Zk9'
    'm?bIDKvrMT<SxpGmMe&0Pf31t72gUu3^0}}z=Vf;sGK64LC)=mWs<&&8+PZoVW%Z&?9IoJtwRtZVUphoN(A=gM$8ZbGRZA%o+Jb^'
    ')RdZp7Q`^s=dv<Usv2n9Rd-}{h3+l7Ir0Q>lbqJHRFt&c$4LFis9RWp98_Mg`&FwKwqJC#y}qe%cM3JO3xY>pgE94cuAp)&1MG^V'
    'uPfSD5Clsk>ocRlRWm;#-fM<`L#iNOghWIpT7Y;|6UUTzlrYD~CYy%^GX{5$^TNMpt#S*<TsA7l4FMbsNi3_@Q?rH>vpDsC#0hV3'
    '8fQMvGJt1K1<ftub5FUZF;rq?)YRP7)RtlLfY9ipoj?d*1*D8-pR9&(YNKl^<JnMC`od#_C3gWgD1@*sqzfa6#qR+kUZuYG%j64U'
    'AZ{cBT<=I$IYUc^upl)+n0q`aVAG771z{d3*qm?yQQ2Uri}oa_JsV(y#2Z=|$JspmMx=bL2<KI6BIC!i1X`tHFVMHOVRqVoEs;J{'
    's%uLeo}4^L1JjhdHGNo+LOdyr_?%RtsnXR_isjRaC#4qSm;qTMv3Oc?A^C0@W!uNJ#l6~HY_%-qi5W{h{ovxRkhiomSvGrVw;+|%'
    'b~pl2T#&On?0q#Z7ro1}C%kM>lnDaJmHLWGYjUM>SLB8v%_Aa|dXSnZ0(hGc168%PTD5vAqR7|E_$@x#dL$v@%eoZzI%-SBTumGX'
    '6Gm7eiq)vZ)rM6X0$R#VSn*T>Kg~mJ4bfi!XAFHLnujofg!HFUp16-whKwO+P*0?I91@BCQlL?0fU=5ub#RmT4r|XTo{Aa;+fnOO'
    'HGNYgk&9*FT2KnYM~R8i7^kK4Kv@o{gMvOT0jp~G!&{I?r_t9&BZBZ{H)a0JjoKP`G>>kH)lbWymq~dcY<nc70gmrCt-Oz8)%I|<'
    ')!RgjES~4dynZNWiESvhJ>J+aXN%0mKgC8=tqD8)==rHpoS+!KP3Mqg(PINn{E_}NYs+azG;Z_XXGI<^9vwDJamAKaVBaNT@no@~'
    'XXBYmAX$Q>FN2{Y!}T77)bAl-8jf7>jv@+8{KzkM0mao@Xc!MTMbbgQ3u$>2sQipoH^c|#WpSp07a^X>BVC=ZxD?>9T<(w4WB1k%'
    '=V7(34{iN0cOg*<2!403GNU0lOmBPk%a_F(e^TRvp{eS~nk}sHDfVO7617yN(t$GMccDfQPy0QTZXEcDq!CN29h9~3oUCaoS*di#'
    '&ArlFqwZzf>Y~-6;RLkOa#r5PbLkFF=7c~>cahG?n5tY1srpxg{^sTmJ-G1I=8K(|dn5XF^aB4!zaEUXb~Xp}>+YbxHP{u;@16_?'
    '8=In(th-MXGa^&)fNOGu!!h#GQ(4r}X!PQMi8?yiTVF3k-P_vVcuBwRzgX+_S?P;W5B^$3<r+OKv91-tD1h)i=;b!hmfcmR(XbQ&'
    '8mVED_FPl#uNDW%$hbI6c3%l((`?r;AG-I_8S@J+%w<JmC(%@2FoTk8dARJhmmI3-b_?Pjb>y~PttLDcz@T7uRxSN4=61&N*LHUG'
    'H#8RQJy6jWJU>%U`DsLRBGAjQhRhdSGnUw~OU2BD9*%Xc+?2t88v$K$85-RN7F$htR<!~O9ZD$EqycDkXk$a8XKxJ<w)rOlqiQqd'
    '4vgB#@;eA*`ZV?H2Zr(Qs#;wb8o-9~YGa72A3)W^n7GkXQT!Ok;jnM7Z8RUz)jxNPDAT6QY@8AjWwSVEIY86@Ae7M|#pC!qS85XG'
    'Fd`YtC!DKkTYD~gy5qQzh5_NQXTKP<65%4sJR?9f*lH!p5`qVWQlRQt3*5VP-%}>5!Q`Ut?utW42nG#=<!PCWs*L>Y9qu$UuRT@9'
    'QOF%a2%{WCR{>B3c=C2jWVBO~1n!s^-)P@h+Be;A08ejbaX`BCg)bR5xweG(by}e&&oAnEijZuaF3dJsCR{7xYE!y8yA_dj5yQpU'
    'x(HVnV(RDP=|U`(a1^~q&PD625HRJ0M<!SiEgwb7Wl?f5LbjmeBgnWA6_-N9$I<XnBzzPF7b2i3?aivUaO!8m1l;2=p@6#bFsF=_'
    'R)dQo6xWQbm~y)wo00Fp1k@;|d-z_>Hoit{kLY*Swc7R*E5ub>x=^xR%zSC-Rh;K>*es{(9nE64Od*%k2{?+db4!X{M35VSu6XYU'
    '1b6VIEu&@Ou9B?&CMSD4vu5#JU`{T*%~?C?P<3P(?n3__<t`qx!4BJg2+nY0C^z&3vW(;oesSW-#E=gDVrjLdNC3mwpc*UBO|0dD'
    't)-r*OCz2rpHyScB-|Fv5f5I$gNYz|@sNR#;$2-!Q@yM!fJOB_*s$?7n3woN{_?Ik@x=9oa=&!2*{jN`&Cz~;t15g&r>4akb(Xtn'
    'UPRKDtcdPj>{8ofFK9EL!sMVT-CFMmda&^wIo|)R$INeSbK_By)2BkuXYNo(*Aa@Yp=gY=>QpE!-$@)Uc)bv5Zm3>{y6^^{+1{W2'
    'TCev-x-|Q<>C(oR=@31`T6^K)iG5skNWM@HM*C;oah#^?)rCYWUSIigU8o(iFLq4i)e*jBBCeANIX=2=;iTMQpNTBBZ#9FXz_ZfP'
    'c|>o9Q~|E4aN0WUVK)PaGoD(0Q%ES63Q*D{SDY;1n59ZwKDP__Cd=x55P<Is^g(IOG*76;_?ZCbc*J#?V7}JxEtMj)E{oN21C@!F'
    'w$IX%+JhGR7k%JqIAn<7=$RPRWZkqmINI&X-dqIWE0_j<)PGr=HN?RBg#J;Z{>3}kY4^Xn?=_|l<x3@&fZkie%;d#}h}pt}!J1;p'
    '`b%sAuuSe3-3s8c?x-gSFe&c8*u?p8pMI{r#Ix^qZ*9Y@bJ%t9EDT<CL5B^d@Ah8tW!KGw#8Ylm80`g%n-S=2SluwR^7?6<;%t>}'
    '-3@iV1*@xd0X?B}0(#wq(ULxB^l9RH!-1;4du!j)$ps~&>@Ytv8u^j<2CP5+IKn>WIVp9C_g;|HJ|DX<SFRo!Ftt7gvwaJ0ENpu='
    'xzL6a^xeP@{Tv3gyJ2Zv<>+ylu~O)M57YEX)nb#kiUqse#6hXcU9?6U_z-tf-7qr(5>4~ja1B=zD@_(gICi6JH?zIey)hPwI|XrJ'
    '>V{HNTr+70ic?ueu^p=xMNZemm56TN-Vj_q)xZ9L2gx#+lKjnmcjMt}!7=Hw394$?YyltPJX}UUdTgo&e|em&m{H^Eq!UNtI*r-+'
    '1@`z<Nc07F`uME29fh`fJ&r;!)saysSq*`bHuPy*DXhWI!ez&0uqKgkhItF)N|y@<__0>)VJ8Yy0(<O!2;!-8-|sybiF5w}Z-Gg}'
    'dkA9>59ZejfJ&YUPRY{VOUb{|4kB=dG120P6n8Gs>jX46@^~4xNT(d+nxuCcx}pY?1~5gdVB<&QV2<Yj@!mKRunuJwOAER%s#OUJ'
    'RsFzLgbVlVYHou~?9dkI-A=_FIC!bm)%OOdZ#)}P$i=I45Lv~kO;wd@3p>F2WC(Zdn2X5U6f7<`F_DS|Cp_j;EH^2Z6^l)jbI6R9'
    'N7{?t75DcToZ7ze^O=*)ClmkH(CzGc3VzvK&oNixl6aHlxo}TuWfx6Fuu!ciEm(wlVxze~nS4g8)KUDv`FP`|(`<OJkK-^&NtPA&'
    '!6ADm`0+-6u%xSwLf4P#x3(!ScIznak^#7**RweeF7G=I?uun`a7`cv3ilh;H{2SRA>r*W?v?M6em6<u#GO*t>CSwZ@2BJ0T{n$y'
    'y2a&2Pdt((-D~vN{Z7hDOcq?bfbaS%FKOYenu-H)mV8@5+pP^MSvtx~1S~np6-divWjv%O4n%V=ROc5~b{T!f6kp4rg$;|=S&gMg'
    'y32&Q77e9hlPXj`P}r_ItKKhDUVUyZo&W!Iq9gmeTHDK-eFKg~2v*uFz9f&A*wmm$)p~uuo8F4h<@~yk?|by>CEjCtrG%XNRThp@'
    'mf~6f;r*a5dnAgT5d5r@N0O|quSnc05Ax}>G9W8&NKHp|33N*?D)DR0C<RpHq%D+q`9XOL1R1lNAoOk3$;USRNvA!<2j-=CONDe!'
    'q$$jU;vk85LyRHm+x!gsq>KtH#olMj*(ST$m|hz-Fyz2xDO#_Ewy3&+1@oYB5ND+;7Nse^tu)ck!Bagv;H|Bf7?j7`+I_8o2fR7C'
    '<4-h)9#(K1Jm#)D4hCe$(V4Nc=-l}~Zee+X'
)


def loadEmbeddedRowModule(encodedSource: str, rowNumber: int) -> types.ModuleType:
    sourceBytes = zlib.decompress(base64.b85decode(encodedSource.encode("ascii")))
    sourceText = sourceBytes.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")

    def replaceRequired(oldText: str, newText: str, description: str) -> None:
        nonlocal sourceText
        if oldText not in sourceText:
            raise RuntimeError(
                f"Could not apply embedded row {rowNumber} modification: {description}."
            )
        sourceText = sourceText.replace(oldText, newText, 1)

    # Use one typography specification throughout every row, including
    # panel letters, titles, axis labels, tick labels, legends, annotations,
    # scientific-notation multipliers, and color-bar text.
    replaceRequired(
        "UNIFIED_FONT_SIZE = 18.0",
        f"UNIFIED_FONT_SIZE = {float(FINAL_FONT_SIZE):.1f}",
        "set the unified font size to 25 points",
    )
    if "PANEL_LABEL_FONT_SIZE = 22.0" in sourceText:
        replaceRequired(
            "PANEL_LABEL_FONT_SIZE = 22.0",
            f"PANEL_LABEL_FONT_SIZE = {float(FINAL_FONT_SIZE):.1f}",
            "set panel-label font size to the unified 25 points",
        )

    if rowNumber == 1:
        replaceRequired(
            "    brainYPositions = {'ASD': 0.76, 'HC': 0.25}\n",
            "    brainYPositions = {'HC': 0.76, 'ASD': 0.25}\n",
            "place HC above ASD in panel a",
        )
        replaceRequired(
            "            addBrainImage(axis, stepImagePaths[groupName, stepNumber], (horizontalPosition, brainYPosition), zoom=0.235)\n",
            "            addBrainImage(axis, stepImagePaths[groupName, stepNumber], (horizontalPosition, brainYPosition), zoom=0.340)\n",
            "further enlarge all propagation brain images in panel a",
        )
        sourceText = "\n".join(
            sourceLine
            for sourceLine in sourceText.splitlines()
            if "axis.text(hierarchyLevel, asdMean" not in sourceLine
            and "axis.text(hierarchyLevel, hcMean" not in sourceLine
        ) + "\n"
        replaceRequired(
            "loc='upper left', bbox_to_anchor=(1.025, 0.985)",
            "loc='lower right', bbox_to_anchor=(0.99, 0.01)",
            "move the ASD/HC line legend inside panel b",
        )
        replaceRequired(
            "axis.text(0.825, brainYPosition if groupName == 'ASD' else brainYPosition - 0.007, groupName,",
            "axis.text(-0.045, brainYPosition if groupName == 'ASD' else brainYPosition - 0.007, groupName,",
            "move the ASD/HC labels to the left of panel a",
        )
        replaceRequired(
            "    axis.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))\n",
            "    axis.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))\n"
            "    bracketBottom = 21.52\n"
            "    bracketTop = 21.58\n"
            "    axis.plot(\n"
            "        (hierarchyLevels[0], hierarchyLevels[0], hierarchyLevels[-1], hierarchyLevels[-1]),\n"
            "        (bracketBottom, bracketTop, bracketTop, bracketBottom),\n"
            "        color=GROUP_COLORS['HC'], linewidth=1.3, solid_capstyle='round',\n"
            "        clip_on=False, zorder=6,\n"
            "    )\n"
            "    axis.text(\n"
            "        np.mean((hierarchyLevels[0], hierarchyLevels[-1])), bracketTop - 0.005, '***',\n"
            "        color='black', ha='center', va='center', fontweight='bold',\n"
            "        clip_on=False, zorder=7,\n"
            "        bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1.2},\n"
            "    )\n",
            "add the L1-L4 significance bracket in panel b",
        )

        # Remove axis-managed headings. Figure-level headings below use one
        # shared top coordinate, so each panel letter and title has the same
        # upper edge even though the two plotting axes have different heights.
        replaceRequired(
            "    axis.set_title('Group-averaged stepwise cortical propagation', x=0.0, y=1.059, ha='left', fontweight='bold', pad=7)\n",
            "",
            "remove the original panel-a title",
        )
        replaceRequired(
            "    addPanelLabel(axis, 'a', horizontalPosition=-0.038, verticalPosition=1.099)\n",
            "",
            "remove the original panel-a letter",
        )
        replaceRequired(
            "        axis.text(-0.045, brainYPosition if groupName == 'ASD' else brainYPosition - 0.007, groupName, color=GROUP_COLORS[groupName], ha='left', va='center', fontweight='bold')\n",
            "",
            "remove the axis-managed ASD/HC labels from panel a",
        )
        replaceRequired(
            "    axis.set_title('Propagation hierarchy across cortical levels (L1–L4)', x=-0.16, y=1.077, ha='left', fontweight='bold', pad=7)\n",
            "",
            "remove the original panel-b title",
        )
        replaceRequired(
            "    axis.set_ylabel('SEC-weighted step centroid', labelpad=10)\n",
            "    axis.set_ylabel('')\n",
            "remove the axis-managed panel-b y-axis title",
        )
        replaceRequired(
            "    axis.tick_params(axis='x', pad=1)\n",
            "    axis.tick_params(axis='x', pad=5)\n",
            "increase the distance between panel-b x ticks and the lower labels/title",
        )
        replaceRequired(
            "    axis.xaxis.labelpad = 0\n",
            "    axis.xaxis.labelpad = 8\n",
            "increase the distance between the panel-b x-axis title and tick labels",
        )
        replaceRequired(
            "    addPanelLabel(axis, 'b', horizontalPosition=-0.235, verticalPosition=1.117)\n",
            "",
            "remove the original panel-b letter",
        )
        replaceRequired(
            "    panelA = figure.add_axes(pixelAxesBounds(117, 92, 947, 430))\n",
            "    panelA = figure.add_axes(pixelAxesBounds(125, 62, 955, 400))\n",
            "shift the entire panel-a propagation block 30 pixels upward while keeping its size unchanged",
        )
        replaceRequired(
            "    panelB = figure.add_axes(pixelAxesBounds(1126, 92, 1646, 353))\n",
            "    panelB = figure.add_axes(pixelAxesBounds(1142, 92, 1662, 353))\n",
            "shift panel b right to increase the gap between the y ticks and the shared y-axis title line",
        )
        replaceRequired(
            "    drawPanelB(panelB, inputTables['metrics'], networkImagePaths)\n"
            "    return figure\n",
            "    drawPanelB(panelB, inputTables['metrics'], networkImagePaths)\n"
            "\n"
            "    headingTop = 1.0 - 20.0 / FIGURE_HEIGHT_PIXELS\n"
            "    figure.text(8.0 / FIGURE_WIDTH_PIXELS, headingTop, 'a', "
            "ha='left', va='top', fontsize=PANEL_LABEL_FONT_SIZE, fontweight='bold')\n"
            "    figure.text(50.0 / FIGURE_WIDTH_PIXELS, headingTop, "
            "'Group-averaged stepwise cortical propagation', "
            "ha='left', va='top', fontweight='bold')\n"
            "    figure.text(1004.0 / FIGURE_WIDTH_PIXELS, headingTop, 'b', "
            "ha='left', va='top', fontsize=PANEL_LABEL_FONT_SIZE, fontweight='bold')\n"
            "    figure.text(1043.0 / FIGURE_WIDTH_PIXELS, headingTop, "
            "'Propagation hierarchy across cortical levels (L1–L4)', "
            "ha='left', va='top', fontweight='bold')\n"
            "    panelAPosition = panelA.get_position()\n"
            "    figure.text(50.0 / FIGURE_WIDTH_PIXELS, panelAPosition.y0 + 0.76 * panelAPosition.height, 'HC', ha='left', va='center')\n"
            "    figure.text(50.0 / FIGURE_WIDTH_PIXELS, panelAPosition.y0 + 0.25 * panelAPosition.height, 'ASD', ha='left', va='center')\n"
            "    panelBPosition = panelB.get_position()\n"
            "    panelBYCenter = 0.5 * (panelBPosition.y0 + panelBPosition.y1)\n"
            "    panelBYTitle = figure.text(1043.0 / FIGURE_WIDTH_PIXELS, panelBYCenter, "
            "'SEC-weighted step centroid', ha='center', va='top', "
            "rotation=90, rotation_mode='anchor')\n"
            "    panelB.tick_params(axis='y', pad=3)\n"
            "    figure.canvas.draw()\n"
            "    renderer = figure.canvas.get_renderer()\n"
            "    tickBounds = [label.get_window_extent(renderer=renderer) for label in panelB.get_yticklabels() if label.get_visible() and label.get_text()]\n"
            "    if tickBounds:\n"
            "        currentTickLeft = min(bounds.x0 for bounds in tickBounds)\n"
            "        titleBounds = panelBYTitle.get_window_extent(renderer=renderer)\n"
            "        desiredTitleRight = currentTickLeft - 5.0\n"
            "        titleShift = (desiredTitleRight - titleBounds.x1) / figure.bbox.width\n"
            "        titleX, titleY = panelBYTitle.get_position()\n"
            "        panelBYTitle.set_position((titleX + titleShift, titleY))\n"
            "    return figure\n",
            "insert top-aligned panel-a and panel-b headings",
        )

    elif rowNumber == 2:
        replaceRequired(
            "    drawGroupCurvePanel(panelDAsd, 'ASD', curveGroupMeans['ASD'], sharedYLimits)\n"
            "    drawGroupCurvePanel(panelDHc, 'HC', curveGroupMeans['HC'], sharedYLimits)\n"
            "    panelDAsd.text(-0.03, 0.93, '$\\\\times 10^{-3}$', transform=panelDAsd.transAxes, ha='right', va='center', clip_on=False)\n",
            "    drawGroupCurvePanel(panelDAsd, 'ASD', curveGroupMeans['ASD'], sharedYLimits)\n"
            "    drawGroupCurvePanel(panelDHc, 'HC', curveGroupMeans['HC'], sharedYLimits)\n"
            "    curveLegendHandles = [\n"
            "        mpl.lines.Line2D([], [], color=SYSTEM_COLORS[systemName], linewidth=DATA_LINE_WIDTH, label=SYSTEM_LEVELS[systemName])\n"
            "        for systemName in SYSTEM_ORDER\n"
            "    ]\n"
            "    panelDHc.legend(\n"
            "        handles=curveLegendHandles, loc='lower right', ncol=2,\n"
            "        handlelength=1.7, columnspacing=0.9, labelspacing=0.35,\n"
            "        borderaxespad=0.25,\n"
            "    )\n"
            "    panelDAsd.text(-0.03, 0.93, '$\\\\times 10^{-3}$', transform=panelDAsd.transAxes, ha='right', va='center', clip_on=False)\n",
            "add the L1-L4 line legend to the final panel-c early-propagation curves",
        )
        replaceRequired(
            "    tableColumnPixelPositions = {'ASD': 650.0, 'HC': 726.0}\n",
            "    tableColumnPixelPositions = {'ASD': 635.0, 'HC': 755.0}\n",
            "increase the spacing between the ASD and HC value columns in panel b",
        )
        replaceRequired(
            "        for groupName in ('ASD', 'HC'):\n",
            "        for groupName in ('ASD', 'HC'):\n",
            "keep ASD drawn before HC in panel c",
        )
        replaceRequired(
            "    groupVerticalOffsets = {'ASD': 0.15, 'HC': -0.15}\n",
            "    groupVerticalOffsets = {'ASD': 0.15, 'HC': -0.15}\n"
            "    displayedScatterHorizontalShift = 0.17\n",
            "define the visual right shift for panel-c scatter points",
        )
        replaceRequired(
            "            axis.scatter(groupValues, verticalCenter + verticalJitter, s=8, color=GROUP_COLORS[groupName], alpha=0.35, linewidth=0, rasterized=True)\n",
            "            axis.scatter(groupValues + displayedScatterHorizontalShift, verticalCenter + verticalJitter, s=8, color=GROUP_COLORS[groupName], alpha=0.35, linewidth=0, rasterized=True)\n",
            "shift panel-c scatter points to the right",
        )
        replaceRequired(
            "            axis.errorbar(meanValue, verticalCenter, xerr=standardError, fmt='o', color='black', ecolor='black', markersize=5.5, elinewidth=ERROR_LINE_WIDTH, capsize=3.5, capthick=ERROR_LINE_WIDTH, zorder=4)\n",
            "            axis.errorbar(meanValue + displayedScatterHorizontalShift, verticalCenter, xerr=standardError, fmt='o', color='black', ecolor='black', markersize=5.5, elinewidth=ERROR_LINE_WIDTH, capsize=3.5, capthick=ERROR_LINE_WIDTH, zorder=4)\n",
            "shift panel-c mean points and error bars to the right",
        )
        replaceRequired(
            "    drawGroupCurvePanel(panelDAsd, 'ASD', curveGroupMeans['ASD'], sharedYLimits)\n"
            "    drawGroupCurvePanel(panelDHc, 'HC', curveGroupMeans['HC'], sharedYLimits)\n",
            "    drawGroupCurvePanel(panelDAsd, 'HC', curveGroupMeans['HC'], sharedYLimits)\n"
            "    drawGroupCurvePanel(panelDHc, 'ASD', curveGroupMeans['ASD'], sharedYLimits)\n",
            "place HC before ASD in panel d",
        )
        replaceRequired(
            "    axis.figure.text(tableColumnPixelPositions['HC'] / FIGURE_WIDTH_PIXELS, 1.0 - 52.0 / FIGURE_HEIGHT_PIXELS, 'HC', color=GROUP_COLORS['HC'], ha='center', va='bottom', fontweight='bold')\n",
            "    axis.figure.text(tableColumnPixelPositions['HC'] / FIGURE_WIDTH_PIXELS, 1.0 - 52.0 / FIGURE_HEIGHT_PIXELS, 'HC', color=GROUP_COLORS['HC'], ha='center', va='bottom', fontweight='bold')\n"
            "\n"
            "    # L1-L4 significance brackets beside the ASD and HC value columns.\n"
            "    # The bracket is deliberately split around the middle so the significance\n"
            "    # symbol sits in the gap, reproducing the reference layout.\n"
            "    bracketTopPixel = tableRowPixelPositions[0]\n"
            "    bracketBottomPixel = tableRowPixelPositions[-1]\n"
            "    bracketMiddlePixel = 0.5 * (tableRowPixelPositions[1] + tableRowPixelPositions[2])\n"
            "    bracketGapHalfHeightPixels = 11.0\n"
            "    bracketCapLengthPixels = 18.0\n"
            "    bracketLineWidth = 1.8\n"
            "    bracketRightPixelPositions = {'ASD': 685.0, 'HC': 805.0}\n"
            "    significanceText = '*'\n"
            "    for groupName in ('ASD', 'HC'):\n"
            "        bracketRightPixel = bracketRightPixelPositions[groupName]\n"
            "        bracketLeftPixel = bracketRightPixel - bracketCapLengthPixels\n"
            "        bracketColor = 'black'\n"
            "        xLeft = bracketLeftPixel / FIGURE_WIDTH_PIXELS\n"
            "        xRight = bracketRightPixel / FIGURE_WIDTH_PIXELS\n"
            "        yTop = 1.0 - bracketTopPixel / FIGURE_HEIGHT_PIXELS\n"
            "        yBottom = 1.0 - bracketBottomPixel / FIGURE_HEIGHT_PIXELS\n"
            "        yUpperEnd = 1.0 - (bracketMiddlePixel - bracketGapHalfHeightPixels) / FIGURE_HEIGHT_PIXELS\n"
            "        yLowerStart = 1.0 - (bracketMiddlePixel + bracketGapHalfHeightPixels) / FIGURE_HEIGHT_PIXELS\n"
            "        for xValues, yValues in (\n"
            "            ((xLeft, xRight), (yTop, yTop)),\n"
            "            ((xRight, xRight), (yTop, yUpperEnd)),\n"
            "            ((xRight, xRight), (yLowerStart, yBottom)),\n"
            "            ((xLeft, xRight), (yBottom, yBottom)),\n"
            "        ):\n"
            "            axis.figure.add_artist(mpl.lines.Line2D(\n"
            "                xValues, yValues, transform=axis.figure.transFigure,\n"
            "                color=bracketColor, linewidth=bracketLineWidth,\n"
            "                solid_capstyle='round', clip_on=False, zorder=8,\n"
            "            ))\n"
            "        axis.figure.text(\n"
            "            (bracketRightPixel + 4.0) / FIGURE_WIDTH_PIXELS,\n"
            "            1.0 - bracketMiddlePixel / FIGURE_HEIGHT_PIXELS,\n"
            "            significanceText, color='black', ha='left', va='center',\n"
            "            fontweight='bold', zorder=9,\n"
            "        )\n",
            "add split L1-L4 significance brackets beside the ASD and HC columns",
        )
        replaceRequired(
            "    axis.set_title('Temporal centroids', loc='left', fontweight='bold', pad=15)\n",
            "",
            "remove the original panel-c title",
        )
        replaceRequired(
            "    axis.set_xlabel('Temporal centroid (step)')",
            "    axis.set_xlabel('SEC-weighted step centroid')",
            "rename the panel-c x-axis",
        )
        replaceRequired(
            "    axis.xaxis.labelpad = 5.5\n",
            "    axis.xaxis.labelpad = 10.5\n",
            "increase the distance between the panel-c x-axis title and tick labels",
        )
        replaceRequired(
            "    addPanelLabel(axis, 'c')\n",
            "",
            "remove the original panel-c letter",
        )
        replaceRequired(
            "    axis.set_yticklabels([SYSTEM_LEVELS[systemName] for systemName in SYSTEM_ORDER])\n",
            "    axis.set_yticklabels(['', '', '', ''])\n",
            "remove the axis-managed L1-L4 tick labels from panel c",
        )
        replaceRequired(
            "    for (tickLabel, systemName) in zip(axis.get_yticklabels(), SYSTEM_ORDER):\n        tickLabel.set_color(SYSTEM_COLORS[systemName])\n",
            "",
            "remove the axis-managed tick-label coloring loop from panel c",
        )
        replaceRequired(
            "    panelDContainer.text(0.0, 0.995, 'd', transform=panelDContainer.transAxes, ha='left', va='top', fontsize=PANEL_LABEL_FONT_SIZE, fontweight='bold')\n",
            "",
            "remove the original panel-d letter",
        )
        replaceRequired(
            "    panelDContainer.text(0.055, 0.993, 'Early propagation dynamics', transform=panelDContainer.transAxes, ha='left', va='top', fontweight='bold')\n",
            "",
            "remove the original panel-d title",
        )
        replaceRequired(
            "    panelDContainer.text(-0.06, 0.5, 'Propagation intensity', transform=panelDContainer.transAxes, ha='center', va='center', rotation=90)\n",
            "",
            "remove the container-managed panel-d y-axis title",
        )
        replaceRequired(
            "    panelDAsd.text(-0.03, 0.93, '$\\\\times 10^{-3}$', transform=panelDAsd.transAxes, ha='right', va='center', clip_on=False)\n",
            "    panelDAsd.text(0.0, 1.015, '$\\\\times 10^{-3}$', transform=panelDAsd.transAxes, ha='left', va='bottom', clip_on=False)\n",
            "move the panel-d scientific-notation multiplier to the top of the y-axis",
        )
        replaceRequired(
            "    panelC = figure.add_axes(pixelAxesBounds(117, 65, PANEL_C_RIGHT_PIXELS, 300))\n",
            "    panelC = figure.add_axes(pixelAxesBounds(130, 88, 700, 300))\n",
            "shift panel c right to provide more separation from the left labels and title line",
        )
        replaceRequired(
            "    panelDAsd = figure.add_axes(pixelAxesBounds(923, 65, 1315, 300))\n",
            "    panelDAsd = figure.add_axes(pixelAxesBounds(1010, 88, 1360, 300))\n",
            "move the ASD propagation panel farther right of the shared title line",
        )
        replaceRequired(
            "    panelDHc = figure.add_axes(pixelAxesBounds(1350, 65, PANEL_D_RIGHT_PIXELS, 300), sharey=panelDAsd)\n",
            "    panelDHc = figure.add_axes(pixelAxesBounds(1400, 88, PANEL_D_RIGHT_PIXELS, 300), sharey=panelDAsd)\n",
            "move the HC propagation panel farther right while preserving its right edge",
        )
        replaceRequired(
            "    panelDContainer.text(0.478, 0.11, 'Step', transform=panelDContainer.transAxes, ha='center', va='top')\n"
            "    return figure\n",
            "    panelDContainer.text(0.478, 0.05513, 'Step', transform=panelDContainer.transAxes, ha='center', va='top')\n"
            "\n"
            "    headingTop = 1.0 - 20.0 / FIGURE_HEIGHT_PIXELS\n"
            "    figure.text(8.0 / FIGURE_WIDTH_PIXELS, headingTop, 'b', "
            "ha='left', va='top', fontsize=PANEL_LABEL_FONT_SIZE, fontweight='bold')\n"
            "    figure.text(50.0 / FIGURE_WIDTH_PIXELS, headingTop, "
            "'SEC-weighted step centroid', ha='left', va='top', fontweight='normal')\n"
            "    figure.text(884.0 / FIGURE_WIDTH_PIXELS, headingTop, 'c', "
            "ha='left', va='top', fontsize=PANEL_LABEL_FONT_SIZE, fontweight='bold')\n"
            "    figure.text(930.0 / FIGURE_WIDTH_PIXELS, headingTop, "
            "'Early propagation dynamics', ha='left', va='top', fontweight='bold')\n"
            "    panelCPosition = panelC.get_position()\n"
            "    panelCL1 = figure.text(50.0 / FIGURE_WIDTH_PIXELS, panelCPosition.y0 + 0.875 * panelCPosition.height, 'L1', ha='left', va='center', color=SYSTEM_COLORS['H1_sensory'])\n"
            "    panelCL2 = figure.text(50.0 / FIGURE_WIDTH_PIXELS, panelCPosition.y0 + 0.625 * panelCPosition.height, 'L2', ha='left', va='center', color=SYSTEM_COLORS['H2_attention'])\n"
            "    panelCL3 = figure.text(50.0 / FIGURE_WIDTH_PIXELS, panelCPosition.y0 + 0.375 * panelCPosition.height, 'L3', ha='left', va='center', color=SYSTEM_COLORS['H3_control'])\n"
            "    panelCL4 = figure.text(50.0 / FIGURE_WIDTH_PIXELS, panelCPosition.y0 + 0.125 * panelCPosition.height, 'L4', ha='left', va='center', color=SYSTEM_COLORS['H4_DMN'])\n"
            "    panelDPosition = panelDAsd.get_position()\n"
            "    panelDYCenter = 0.5 * (panelDPosition.y0 + panelDPosition.y1)\n"
            "    panelDYTitle = figure.text(930.0 / FIGURE_WIDTH_PIXELS, panelDYCenter, "
            "'Propagation intensity', ha='center', va='top', "
            "rotation=90, rotation_mode='anchor')\n"
            "    panelDAsd.tick_params(axis='y', pad=3)\n"
            "    figure.canvas.draw()\n"
            "    renderer = figure.canvas.get_renderer()\n"
            "    panelCLabelBounds = [artist.get_window_extent(renderer=renderer) for artist in (panelCL1, panelCL2, panelCL3, panelCL4)]\n"
            "    if panelCLabelBounds:\n"
            "        targetPanelCLeft = max(bounds.x1 for bounds in panelCLabelBounds) + 18.0\n"
            "        panelCPosition = panelC.get_position()\n"
            "        currentPanelCLeft = panelCPosition.x0 * figure.bbox.width\n"
            "        panelCShift = (targetPanelCLeft - currentPanelCLeft) / figure.bbox.width\n"
            "        panelC.set_position((panelCPosition.x0 + panelCShift, panelCPosition.y0, panelCPosition.width - panelCShift, panelCPosition.height))\n"
            "    figure.canvas.draw()\n"
            "    renderer = figure.canvas.get_renderer()\n"
            "    titleBounds = panelDYTitle.get_window_extent(renderer=renderer)\n"
            "    desiredTitleLeft = (930.0 / FIGURE_WIDTH_PIXELS) * figure.bbox.width\n"
            "    titleShift = (desiredTitleLeft - titleBounds.x0) / figure.bbox.width\n"
            "    titleX, titleY = panelDYTitle.get_position()\n"
            "    panelDYTitle.set_position((titleX + titleShift, titleY))\n"
            "    return figure\n",
            "insert top-aligned panel-c and panel-d headings",
        )
        # Rebuild panel d at its new aspect ratio before the final page is
        # assembled. This changes the plotting axes themselves, rather than
        # stretching a rendered raster panel.
        replaceRequired(
            "    panelDContainer = figure.add_axes(pixelAxesBounds(PANEL_D_LEFT_PIXELS, 20, PANEL_D_RIGHT_PIXELS, 385))\n",
            "    panelDContainer = figure.add_axes(pixelAxesBounds(PANEL_D_LEFT_PIXELS, 15, 1600, 405))\n",
            "set a narrower and taller panel-d container",
        )
        replaceRequired(
            "    panelDAsd = figure.add_axes(pixelAxesBounds(1010, 88, 1360, 300))\n",
            "    panelDAsd = figure.add_axes(pixelAxesBounds(1000, 115, 1300, 355))\n",
            "set a narrower and taller ASD axis in panel d",
        )
        replaceRequired(
            "    panelDHc = figure.add_axes(pixelAxesBounds(1400, 88, PANEL_D_RIGHT_PIXELS, 300), sharey=panelDAsd)\n",
            "    panelDHc = figure.add_axes(pixelAxesBounds(1350, 115, 1650, 355), sharey=panelDAsd)\n",
            "set a narrower and taller HC axis in panel d",
        )

    elif rowNumber == 3:
        replaceRequired(
            "    axis.set_title(title, x=0.08, ha='center', fontweight='bold', pad=5, linespacing=1.05)\n",
            "",
            "remove axis-managed raincloud titles",
        )
        replaceRequired(
            "    axis.set_ylabel(yLabel)\n",
            "    axis.set_ylabel('')\n",
            "remove axis-managed raincloud y-axis titles",
        )
        replaceRequired(
            "    panelContainer.text(firstPanelLabelLeft, 0.95, 'e', transform=panelContainer.transAxes, ha='left', va='top', fontsize=PANEL_LABEL_FONT_SIZE, fontweight='bold')\n",
            "",
            "remove the original panel-e letter",
        )
        replaceRequired(
            "    panelContainer.text(supplementaryGroupLeft, 0.95, 'f', transform=panelContainer.transAxes, ha='left', va='top', fontsize=PANEL_LABEL_FONT_SIZE, fontweight='bold')\n",
            "",
            "remove the original panel-f letter",
        )
        replaceRequired(
            "    raincloudBounds = ((127, 82, 327, 322), (468, 82, 668, 322), (809, 82, 1009, 322), (1150, 82, 1350, 322), (1491, 82, 1691, 322))\n",
            "    raincloudBounds = ((140, 70, 340, 355), (481, 70, 681, 355), (822, 70, 1022, 355), (1163, 70, 1363, 355), (1504, 70, 1704, 355))\n",
            "shift the raincloud panels right to open space beside the left title line",
        )
        replaceRequired(
            "        drawRaincloudPanel(axis, inputTables['metrics'], valueColumn, raincloudTitles[systemName], 'Early SEC-slope', '$\\\\times 10^{-5}$', correctedPValue, displayScale=100000.0)\n"
            "    return figure\n",
            "        drawRaincloudPanel(axis, inputTables['metrics'], valueColumn, raincloudTitles[systemName], 'Early SEC-slope', '$\\\\times 10^{-5}$', correctedPValue, displayScale=100000.0)\n"
            "\n"
            "    headingTop = 1.0 - 17.4 / FIGURE_HEIGHT_PIXELS\n"
            "    figure.text(8.0 / FIGURE_WIDTH_PIXELS, headingTop, 'd', "
            "ha='left', va='top', fontsize=PANEL_LABEL_FONT_SIZE, fontweight='bold')\n"
            "    figure.text(349.0 / FIGURE_WIDTH_PIXELS, headingTop, 'e', "
            "ha='left', va='top', fontsize=PANEL_LABEL_FONT_SIZE, fontweight='bold')\n"
            "    headingLefts = tuple(\n"
            "        (leftPixel - 75.0) / FIGURE_WIDTH_PIXELS\n"
            "        for leftPixel, _, _, _ in raincloudBounds\n"
            "    )\n"
            "    headingTexts = (\n"
            "        'L4–L1 contrast\\nEarly SEC slope',\n"
            "        'L1 sensory\\nEarly SEC slope',\n"
            "        'L2 attention\\nEarly SEC slope',\n"
            "        'L3 control\\nEarly SEC slope',\n"
            "        'L4 DMN\\nEarly SEC slope',\n"
            "    )\n"
            "    for headingLeft, headingText in zip(headingLefts, headingTexts):\n"
            "        figure.text(\n"
            "            headingLeft, headingTop, headingText,\n"
            "            ha='left', va='top', fontweight='bold', linespacing=1.05,\n"
            "        )\n"
            "    yAxisTitleTexts = (\n"
            "        'Slope contrast',\n"
            "        'Early SEC-slope',\n"
            "        'Early SEC-slope',\n"
            "        'Early SEC-slope',\n"
            "        'Early SEC-slope',\n"
            "    )\n"
            "    yAxisTitleArtists = []\n"
            "    for headingLeft, yAxisTitleText, rainAxis in zip(headingLefts, yAxisTitleTexts, raincloudAxes):\n"
            "        rainAxisPosition = rainAxis.get_position()\n"
            "        yAxisTitleY = 0.5 * (rainAxisPosition.y0 + rainAxisPosition.y1)\n"
            "        yAxisTitleArtists.append(figure.text(\n"
            "            headingLeft, yAxisTitleY, yAxisTitleText,\n"
            "            ha='center', va='top', rotation=90, rotation_mode='anchor',\n"
            "        ))\n"
            "        rainAxis.tick_params(axis='y', pad=3)\n"
            "    figure.canvas.draw()\n"
            "    renderer = figure.canvas.get_renderer()\n"
            "    for rainAxis, yAxisTitleArtist in zip(raincloudAxes, yAxisTitleArtists):\n"
            "        offsetText = rainAxis.yaxis.get_offset_text()\n"
            "        offsetText.set_x(0.0)\n"
            "        offsetText.set_y(1.015)\n"
            "        offsetText.set_ha('left')\n"
            "        offsetText.set_va('bottom')\n"
            "        tickBounds = [label.get_window_extent(renderer=renderer) for label in rainAxis.get_yticklabels() if label.get_visible() and label.get_text()]\n"
            "        if not tickBounds:\n"
            "            continue\n"
            "        currentTickLeft = min(bounds.x0 for bounds in tickBounds)\n"
            "        titleBounds = yAxisTitleArtist.get_window_extent(renderer=renderer)\n"
            "        desiredTitleRight = currentTickLeft - 5.0\n"
            "        titleShift = (desiredTitleRight - titleBounds.x1) / figure.bbox.width\n"
            "        titleX, titleY = yAxisTitleArtist.get_position()\n"
            "        yAxisTitleArtist.set_position((titleX + titleShift, titleY))\n"
            "    return figure\n",
            "insert headings and y-axis titles with identical visual left edges",
        )

    elif rowNumber == 4:
        replaceRequired(
            "        slopeColumns = [\n"
            "            f\"ROI_{roiIndex:03d}_early_slope_1_10\"\n"
            "            for roiIndex in systemRoiIndices\n"
            "        ]\n"
            "        for groupName in GROUP_ORDER:\n"
            "            groupRows = slopeTable.loc[slopeTable[\"Group\"].eq(groupName), slopeColumns]\n"
            "            groupLevelMean = float(groupRows.to_numpy(float).mean())\n"
            "            groupValues = np.full(N_PARCELS, np.nan, dtype=float)\n"
            "            groupValues[systemRoiIndices] = groupLevelMean\n"
            "            groupSlopeMaps[(groupName, systemName)] = groupValues\n",
            "        betaAsdMinusHc = float(resultRow[\"Beta_ASD_minus_HC\"].iloc[0])\n"
            "        for groupName in GROUP_ORDER:\n"
            "            betaValues = np.full(N_PARCELS, np.nan, dtype=float)\n"
            "            betaValues[systemRoiIndices] = betaAsdMinusHc\n"
            "            groupSlopeMaps[(groupName, systemName)] = betaValues\n",
            "render the existing vertically paired HC and ASD brain slots with the adjusted ASD-minus-HC beta value",
        )
        replaceRequired(
            'def cropLateralPair(imagePath: Path) -> None:\n',
            'def extractVisibleBrainView(\n    sourceImage: Image.Image,\n    paddingPixels: int = 8,\n) -> Image.Image:\n    whiteBackground = Image.new("RGB", sourceImage.size, "white")\n    differenceImage = ImageChops.difference(\n        sourceImage,\n        whiteBackground,\n    ).convert("L")\n    visibleMask = differenceImage.point(\n        lambda pixelValue: 255 if pixelValue > 4 else 0\n    )\n    visibleBounds = visibleMask.getbbox()\n    if visibleBounds is None:\n        raise RuntimeError("BrainSpace produced an empty cortical view.")\n    left, upper, right, lower = visibleBounds\n    cropBounds = (\n        max(0, left - paddingPixels),\n        max(0, upper - paddingPixels),\n        min(sourceImage.width, right + paddingPixels),\n        min(sourceImage.height, lower + paddingPixels),\n    )\n    return sourceImage.crop(cropBounds)\n\n\ndef fitBrainViewToCanvas(\n    sourceImage: Image.Image,\n    canvasWidth: int = 250,\n    canvasHeight: int = 228,\n) -> Image.Image:\n    widthScale = canvasWidth / sourceImage.width\n    heightScale = canvasHeight / sourceImage.height\n    uniformScale = min(widthScale, heightScale)\n    resizedWidth = max(1, round(sourceImage.width * uniformScale))\n    resizedHeight = max(1, round(sourceImage.height * uniformScale))\n    resizedImage = sourceImage.resize(\n        (resizedWidth, resizedHeight),\n        Image.Resampling.LANCZOS,\n    )\n    normalizedCanvas = Image.new(\n        "RGB",\n        (canvasWidth, canvasHeight),\n        "white",\n    )\n    pastePosition = (\n        (canvasWidth - resizedWidth) // 2,\n        (canvasHeight - resizedHeight) // 2,\n    )\n    normalizedCanvas.paste(resizedImage, pastePosition)\n    return normalizedCanvas\n\n\ndef assembleNormalizedBrainGrid(\n    sourceViews: list[Image.Image],\n    imagePath: Path,\n    nRows: int,\n    nCols: int,\n    canvasWidth: int = 250,\n    canvasHeight: int = 228,\n    horizontalGapPixels: int = 20,\n    verticalGapPixels: int = 16,\n) -> None:\n    normalizedViews = [\n        fitBrainViewToCanvas(extractVisibleBrainView(viewImage), canvasWidth, canvasHeight)\n        for viewImage in sourceViews\n    ]\n    combinedWidth = nCols * canvasWidth + (nCols - 1) * horizontalGapPixels\n    combinedHeight = nRows * canvasHeight + (nRows - 1) * verticalGapPixels\n    combinedImage = Image.new("RGB", (combinedWidth, combinedHeight), "white")\n    for viewIndex, viewImage in enumerate(normalizedViews):\n        rowIndex = viewIndex // nCols\n        columnIndex = viewIndex % nCols\n        horizontalPosition = columnIndex * (canvasWidth + horizontalGapPixels)\n        verticalPosition = rowIndex * (canvasHeight + verticalGapPixels)\n        combinedImage.paste(viewImage, (horizontalPosition, verticalPosition))\n    combinedImage.save(imagePath)\n\n\ndef cropLateralPair(imagePath: Path) -> None:\n',
            'insert shared helpers so every panel-g and panel-h brain fills one identical visible brain canvas',
        )
        replaceRequired(
            '    paddingPixels = 8\n    hemisphereCrops: list[Image.Image] = []\n    halfWidth = lateralPair.width // 2\n    for hemisphereIndex in range(2):\n        hemisphereImage = lateralPair.crop(\n            (\n                hemisphereIndex * halfWidth,\n                0,\n                (hemisphereIndex + 1) * halfWidth,\n                lateralPair.height,\n            )\n        )\n        whiteBackground = Image.new("RGB", hemisphereImage.size, "white")\n        differenceImage = ImageChops.difference(\n            hemisphereImage,\n            whiteBackground,\n        ).convert("L")\n        visibleMask = differenceImage.point(\n            lambda pixelValue: 255 if pixelValue > 4 else 0\n        )\n        visibleBounds = visibleMask.getbbox()\n        if visibleBounds is None:\n            raise RuntimeError("BrainSpace produced an empty hemisphere view.")\n        left, upper, right, lower = visibleBounds\n        cropBounds = (\n            max(0, left - paddingPixels),\n            max(0, upper - paddingPixels),\n            min(hemisphereImage.width, right + paddingPixels),\n            min(hemisphereImage.height, lower + paddingPixels),\n        )\n        hemisphereCrops.append(hemisphereImage.crop(cropBounds))\n\n    interHemisphereGapPixels = 10\n    combinedHeight = max(image.height for image in hemisphereCrops)\n    combinedWidth = (\n        sum(image.width for image in hemisphereCrops)\n        + interHemisphereGapPixels\n    )\n    combinedImage = Image.new("RGB", (combinedWidth, combinedHeight), "white")\n    horizontalPosition = 0\n    for hemisphereImage in hemisphereCrops:\n        verticalPosition = (combinedHeight - hemisphereImage.height) // 2\n        combinedImage.paste(\n            hemisphereImage,\n            (horizontalPosition, verticalPosition),\n        )\n        horizontalPosition += hemisphereImage.width + interHemisphereGapPixels\n    combinedImage.save(imagePath)\n',
            '    hemisphereViews: list[Image.Image] = []\n    halfWidth = lateralPair.width // 2\n    for hemisphereIndex in range(2):\n        hemisphereViews.append(\n            lateralPair.crop(\n                (\n                    hemisphereIndex * halfWidth,\n                    0,\n                    (hemisphereIndex + 1) * halfWidth,\n                    lateralPair.height,\n                )\n            )\n        )\n\n    assembleNormalizedBrainGrid(\n        hemisphereViews,\n        imagePath,\n        nRows=1,\n        nCols=2,\n    )\n',
            'render panel-g brains through the shared fixed-size brain-grid helper',
        )
        replaceRequired(
            '    paddingPixels = 8\n    viewCrops: list[Image.Image] = []\n    for rowIndex in range(2):\n        for columnIndex in range(2):\n            viewImage = sourceImage.crop(\n                (\n                    columnIndex * halfWidth,\n                    rowIndex * halfHeight,\n                    (columnIndex + 1) * halfWidth,\n                    (rowIndex + 1) * halfHeight,\n                )\n            )\n            whiteBackground = Image.new("RGB", viewImage.size, "white")\n            differenceImage = ImageChops.difference(\n                viewImage,\n                whiteBackground,\n            ).convert("L")\n            visibleMask = differenceImage.point(\n                lambda pixelValue: 255 if pixelValue > 4 else 0\n            )\n            visibleBounds = visibleMask.getbbox()\n            if visibleBounds is None:\n                raise RuntimeError("BrainSpace produced an empty cortical view.")\n            left, upper, right, lower = visibleBounds\n            viewCrops.append(\n                viewImage.crop(\n                    (\n                        max(0, left - paddingPixels),\n                        max(0, upper - paddingPixels),\n                        min(viewImage.width, right + paddingPixels),\n                        min(viewImage.height, lower + paddingPixels),\n                    )\n                )\n            )\n\n    columnWidths = (\n        max(viewCrops[0].width, viewCrops[2].width),\n        max(viewCrops[1].width, viewCrops[3].width),\n    )\n    rowHeights = (\n        max(viewCrops[0].height, viewCrops[1].height),\n        max(viewCrops[2].height, viewCrops[3].height),\n    )\n    horizontalGapPixels = 10\n    verticalGapPixels = 8\n    combinedImage = Image.new(\n        "RGB",\n        (\n            sum(columnWidths) + horizontalGapPixels,\n            sum(rowHeights) + verticalGapPixels,\n        ),\n        "white",\n    )\n    for viewIndex, viewImage in enumerate(viewCrops):\n        rowIndex = viewIndex // 2\n        columnIndex = viewIndex % 2\n        horizontalPosition = (\n            0\n            if columnIndex == 0\n            else columnWidths[0] + horizontalGapPixels\n        )\n        verticalPosition = (\n            0\n            if rowIndex == 0\n            else rowHeights[0] + verticalGapPixels\n        )\n        horizontalPosition += (\n            columnWidths[columnIndex] - viewImage.width\n        ) // 2\n        verticalPosition += (\n            rowHeights[rowIndex] - viewImage.height\n        ) // 2\n        combinedImage.paste(\n            viewImage,\n            (horizontalPosition, verticalPosition),\n        )\n    combinedImage.save(imagePath)\n',
            '    viewImages: list[Image.Image] = []\n    for rowIndex in range(2):\n        for columnIndex in range(2):\n            viewImages.append(\n                sourceImage.crop(\n                    (\n                        columnIndex * halfWidth,\n                        rowIndex * halfHeight,\n                        (columnIndex + 1) * halfWidth,\n                        (rowIndex + 1) * halfHeight,\n                    )\n                )\n            )\n\n    assembleNormalizedBrainGrid(\n        viewImages,\n        imagePath,\n        nRows=2,\n        nCols=2,\n    )\n',
            'render panel-h brains through the shared fixed-size brain-grid helper',
        )
        replaceRequired(
            'def createFigure(\n    slopeImagePaths: dict[tuple[str, str], Path],\n    pValueImagePaths: dict[str, Path],\n    slopeColorMaximum: float,\n    pColorMaximum: float,\n) -> plt.Figure:\n    figure = plt.figure(\n        figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES),\n        facecolor="white",\n        dpi=RASTER_DPI,\n    )\n    tileWidth = 260\n    tileHeight = 118\n    tileGap = 10\n    startLeft = 90\n    upperBottom = 180\n    lowerBottom = 84\n\n    figure.text(\n        8 / FIGURE_WIDTH_PIXELS,\n        365 / FIGURE_HEIGHT_PIXELS,\n        "g",\n        ha="left",\n        va="center",\n        fontsize=UNIFIED_FONT_SIZE,\n        fontweight="bold",\n    )\n    figure.text(\n        1300 / FIGURE_WIDTH_PIXELS,\n        365 / FIGURE_HEIGHT_PIXELS,\n        "h",\n        ha="left",\n        va="center",\n        fontsize=UNIFIED_FONT_SIZE,\n        fontweight="bold",\n    )\n    figure.text(\n        50 / FIGURE_WIDTH_PIXELS,\n        365 / FIGURE_HEIGHT_PIXELS,\n        "Mean Early SEC slope",\n        ha="left",\n        va="center",\n        fontweight="bold",\n    )\n    figure.text(\n        1350 / FIGURE_WIDTH_PIXELS,\n        365 / FIGURE_HEIGHT_PIXELS,\n        "FDR-corrected p value",\n        ha="left",\n        va="center",\n        fontweight="bold",\n    )\n\n    for systemIndex, systemName in enumerate(SYSTEM_ORDER):\n        tileLeft = startLeft + systemIndex * (tileWidth + tileGap)\n        addImageAxis(\n            figure,\n            slopeImagePaths[("HC", systemName)],\n            tileLeft,\n            upperBottom,\n            tileWidth,\n            tileHeight,\n        )\n        addImageAxis(\n            figure,\n            slopeImagePaths[("ASD", systemName)],\n            tileLeft,\n            lowerBottom,\n            tileWidth,\n            tileHeight,\n        )\n        figure.text(\n            (tileLeft + tileWidth / 2) / FIGURE_WIDTH_PIXELS,\n            310 / FIGURE_HEIGHT_PIXELS,\n            SYSTEM_LEVELS[systemName],\n            ha="center",\n            va="bottom",\n            fontweight="bold",\n        )\n\n    figure.text(\n        50 / FIGURE_WIDTH_PIXELS,\n        (upperBottom + tileHeight / 2) / FIGURE_HEIGHT_PIXELS,\n        "HC",\n        ha="left",\n        va="center",\n        color="black",\n        fontweight="bold",\n    )\n    figure.text(\n        50 / FIGURE_WIDTH_PIXELS,\n        (lowerBottom + tileHeight / 2) / FIGURE_HEIGHT_PIXELS,\n        "ASD",\n        ha="left",\n        va="center",\n        color="black",\n        fontweight="bold",\n    )\n\n    addImageAxis(\n        figure,\n        pValueImagePaths["FDR p"],\n        1380,\n        75,\n        360,\n        236,\n    )\n\n    slopeColorMap = LinearSegmentedColormap.from_list(\n        "earlySecSlopeFigure",\n        ("#2166AC", "#67A9CF", "#F7F7F7", "#EF8A62", "#B2182B"),\n        N=256,\n    )\n    slopeColorAxis = figure.add_axes((0.176, 0.16, 0.390, 0.035))\n    slopeColorBar = mpl.colorbar.ColorbarBase(\n        slopeColorAxis,\n        cmap=slopeColorMap,\n        norm=Normalize(-slopeColorMaximum, slopeColorMaximum),\n        orientation="horizontal",\n    )\n    slopeColorBar.set_label("Mean Early SEC slope", labelpad=2)\n    slopeColorBar.ax.xaxis.label.set_size(UNIFIED_FONT_SIZE)\n    slopeTickFormatter = mpl.ticker.ScalarFormatter(useMathText=True)\n    slopeTickFormatter.set_powerlimits((-5, -5))\n    slopeColorBar.ax.xaxis.set_major_formatter(slopeTickFormatter)\n    slopeColorBar.update_ticks()\n    slopeColorBar.ax.tick_params(length=3, pad=2)\n    figure.canvas.draw()\n    slopeColorBar.ax.xaxis.get_offset_text().set_visible(False)\n    slopeColorBar.ax.text(\n        1.0,\n        -1.55,\n        r"$\\times 10^{-5}$",\n        transform=slopeColorBar.ax.transAxes,\n        ha="right",\n        va="top",\n        fontsize=UNIFIED_FONT_SIZE,\n        clip_on=False,\n    )\n\n    pValueColorMap = LinearSegmentedColormap.from_list(\n        "negativeLog10FdrPFigure",\n        ("#FFF7EC", "#FEC44F", "#FC8D59", "#D7301F", "#7F0000"),\n        N=256,\n    )\n    pColorAxis = figure.add_axes((0.800, 0.16, 0.130, 0.035))\n    pColorBar = mpl.colorbar.ColorbarBase(\n        pColorAxis,\n        cmap=pValueColorMap,\n        norm=Normalize(0.0, pColorMaximum),\n        orientation="horizontal",\n    )\n    pColorBar.set_ticks((0.0, 1.0, 2.0, 3.0))\n    pColorBar.set_label(r"$-\\log_{10}(p_{\\mathrm{FDR}})$", labelpad=2)\n    pColorBar.ax.xaxis.label.set_size(UNIFIED_FONT_SIZE)\n    pColorBar.ax.tick_params(length=3, pad=2)\n\n    # Enforce one font family and one font size for every text artist,\n    # including panel labels, color-bar labels, tick labels, and annotations.\n    for textArtist in figure.findobj(match=mpl.text.Text):\n        textArtist.set_fontfamily(FONT_FAMILY)\n        textArtist.set_fontsize(UNIFIED_FONT_SIZE)\n\n    return figure\n',
            'def extractCompositeBrainTiles(\n    imagePath: Path,\n    nRows: int,\n    nCols: int,\n    tileSourceWidth: int = 250,\n    tileSourceHeight: int = 228,\n    horizontalGapPixels: int = 20,\n    verticalGapPixels: int = 16,\n) -> list[Image.Image]:\n    compositeImage = Image.open(imagePath).convert("RGB")\n    tiles: list[Image.Image] = []\n    for rowIndex in range(nRows):\n        for columnIndex in range(nCols):\n            left = columnIndex * (tileSourceWidth + horizontalGapPixels)\n            upper = rowIndex * (tileSourceHeight + verticalGapPixels)\n            rawTile = compositeImage.crop(\n                (\n                    left,\n                    upper,\n                    left + tileSourceWidth,\n                    upper + tileSourceHeight,\n                )\n            )\n            tiles.append(\n                shrinkBrainTileToUniformDisplay(\n                    rawTile,\n                    tileSourceWidth,\n                    tileSourceHeight,\n                )\n            )\n    return tiles\n\n\ndef shrinkBrainTileToUniformDisplay(\n    tileImage: Image.Image,\n    canvasWidth: int = 250,\n    canvasHeight: int = 228,\n    targetBrainWidth: int = 186,\n    targetBrainHeight: int = 156,\n) -> Image.Image:\n    visibleBrain = extractVisibleBrainView(tileImage, paddingPixels=0)\n\n    widthScale = targetBrainWidth / visibleBrain.width\n    heightScale = targetBrainHeight / visibleBrain.height\n    uniformScale = min(widthScale, heightScale)\n    resizedWidth = max(1, round(visibleBrain.width * uniformScale))\n    resizedHeight = max(1, round(visibleBrain.height * uniformScale))\n    resizedBrain = visibleBrain.resize(\n        (resizedWidth, resizedHeight),\n        Image.Resampling.LANCZOS,\n    )\n\n    normalizedCanvas = Image.new(\n        "RGB",\n        (canvasWidth, canvasHeight),\n        "white",\n    )\n    pastePosition = (\n        (canvasWidth - resizedWidth) // 2,\n        (canvasHeight - resizedHeight) // 2,\n    )\n    normalizedCanvas.paste(resizedBrain, pastePosition)\n    return normalizedCanvas\n\n\ndef addTileImageAxis(\n    figure: plt.Figure,\n    tileImage: Image.Image,\n    leftPixel: float,\n    bottomPixel: float,\n    widthPixel: float,\n    heightPixel: float,\n) -> plt.Axes:\n    axis = figure.add_axes(\n        (\n            leftPixel / FIGURE_WIDTH_PIXELS,\n            bottomPixel / FIGURE_HEIGHT_PIXELS,\n            widthPixel / FIGURE_WIDTH_PIXELS,\n            heightPixel / FIGURE_HEIGHT_PIXELS,\n        )\n    )\n    axis.imshow(tileImage)\n    axis.set_axis_off()\n    return axis\n\n\ndef createFigure(\n    slopeImagePaths: dict[tuple[str, str], Path],\n    pValueImagePaths: dict[str, Path],\n    slopeColorMaximum: float,\n    pColorMaximum: float,\n) -> plt.Figure:\n    figure = plt.figure(\n        figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES),\n        facecolor="white",\n        dpi=RASTER_DPI,\n    )\n    brainViewWidth = 148\n    tileHeight = 140\n    brainViewHeight = tileHeight\n    withinSystemGap = 8\n    systemGap = 6\n    startLeft = 82\n    upperBottom = 160\n    lowerBottom = 56\n    systemCompositeWidth = 2 * brainViewWidth + withinSystemGap\n    hPanelLabelLeft = 1363\n    hTitleLeft = 1403\n    hBlockLeft = 1391\n\n    figure.text(\n        8 / FIGURE_WIDTH_PIXELS,\n        365 / FIGURE_HEIGHT_PIXELS,\n        "g",\n        ha="left",\n        va="center",\n        fontsize=UNIFIED_FONT_SIZE,\n        fontweight="bold",\n    )\n    figure.text(\n        hPanelLabelLeft / FIGURE_WIDTH_PIXELS,\n        365 / FIGURE_HEIGHT_PIXELS,\n        "h",\n        ha="left",\n        va="center",\n        fontsize=UNIFIED_FONT_SIZE,\n        fontweight="bold",\n    )\n    figure.text(\n        50 / FIGURE_WIDTH_PIXELS,\n        365 / FIGURE_HEIGHT_PIXELS,\n        "Mean Early SEC slope",\n        ha="left",\n        va="center",\n        fontweight="bold",\n    )\n    figure.text(\n        hTitleLeft / FIGURE_WIDTH_PIXELS,\n        365 / FIGURE_HEIGHT_PIXELS,\n        "FDR-corrected p value",\n        ha="left",\n        va="center",\n        fontweight="bold",\n    )\n\n    for systemIndex, systemName in enumerate(SYSTEM_ORDER):\n        groupLeft = startLeft + systemIndex * (systemCompositeWidth + systemGap)\n        hcTiles = extractCompositeBrainTiles(\n            slopeImagePaths[("HC", systemName)],\n            nRows=1,\n            nCols=2,\n        )\n        asdTiles = extractCompositeBrainTiles(\n            slopeImagePaths[("ASD", systemName)],\n            nRows=1,\n            nCols=2,\n        )\n        for tileIndex, tileImage in enumerate(hcTiles):\n            tileLeft = groupLeft + tileIndex * (brainViewWidth + withinSystemGap)\n            addTileImageAxis(\n                figure,\n                tileImage,\n                tileLeft,\n                upperBottom,\n                brainViewWidth,\n                brainViewHeight,\n            )\n        for tileIndex, tileImage in enumerate(asdTiles):\n            tileLeft = groupLeft + tileIndex * (brainViewWidth + withinSystemGap)\n            addTileImageAxis(\n                figure,\n                tileImage,\n                tileLeft,\n                lowerBottom,\n                brainViewWidth,\n                brainViewHeight,\n            )\n        figure.text(\n            (groupLeft + systemCompositeWidth / 2) / FIGURE_WIDTH_PIXELS,\n            310 / FIGURE_HEIGHT_PIXELS,\n            SYSTEM_LEVELS[systemName],\n            ha="center",\n            va="bottom",\n            fontweight="bold",\n        )\n\n    figure.text(\n        50 / FIGURE_WIDTH_PIXELS,\n        (upperBottom + brainViewHeight / 2) / FIGURE_HEIGHT_PIXELS,\n        "HC",\n        ha="left",\n        va="center",\n        color="black",\n        fontweight="bold",\n    )\n    figure.text(\n        50 / FIGURE_WIDTH_PIXELS,\n        (lowerBottom + brainViewHeight / 2) / FIGURE_HEIGHT_PIXELS,\n        "ASD",\n        ha="left",\n        va="center",\n        color="black",\n        fontweight="bold",\n    )\n\n    pTiles = extractCompositeBrainTiles(\n        pValueImagePaths["FDR p"],\n        nRows=2,\n        nCols=2,\n    )\n    pLeft = hBlockLeft\n    pTopBottom = upperBottom\n    pBottomBottom = lowerBottom\n    pWithinColumnGap = withinSystemGap\n    pTilePositions = (\n        (pLeft, pTopBottom),\n        (pLeft + brainViewWidth + pWithinColumnGap, pTopBottom),\n        (pLeft, pBottomBottom),\n        (pLeft + brainViewWidth + pWithinColumnGap, pBottomBottom),\n    )\n    for tileImage, (tileLeft, tileBottom) in zip(pTiles, pTilePositions):\n        addTileImageAxis(\n            figure,\n            tileImage,\n            tileLeft,\n            tileBottom,\n            brainViewWidth,\n            brainViewHeight,\n        )\n\n    slopeColorMap = LinearSegmentedColormap.from_list(\n        "earlySecSlopeFigure",\n        ("#2166AC", "#67A9CF", "#F7F7F7", "#EF8A62", "#B2182B"),\n        N=256,\n    )\n    slopeColorAxis = figure.add_axes((0.150, 0.135, 0.460, 0.035))\n    slopeColorBar = mpl.colorbar.ColorbarBase(\n        slopeColorAxis,\n        cmap=slopeColorMap,\n        norm=Normalize(-slopeColorMaximum, slopeColorMaximum),\n        orientation="horizontal",\n    )\n    slopeColorBar.set_label("Mean Early SEC slope", labelpad=2)\n    slopeColorBar.ax.xaxis.label.set_size(UNIFIED_FONT_SIZE)\n    slopeTickFormatter = mpl.ticker.ScalarFormatter(useMathText=True)\n    slopeTickFormatter.set_powerlimits((-5, -5))\n    slopeColorBar.ax.xaxis.set_major_formatter(slopeTickFormatter)\n    slopeColorBar.update_ticks()\n    slopeColorBar.ax.tick_params(length=3, pad=2)\n    figure.canvas.draw()\n    slopeColorBar.ax.xaxis.get_offset_text().set_visible(False)\n    slopeColorBar.ax.text(\n        1.0,\n        -1.20,\n        r"$\\times 10^{-5}$",\n        transform=slopeColorBar.ax.transAxes,\n        ha="right",\n        va="top",\n        fontsize=UNIFIED_FONT_SIZE,\n        clip_on=False,\n    )\n\n    pValueColorMap = LinearSegmentedColormap.from_list(\n        "negativeLog10FdrPFigure",\n        ("#FFF7EC", "#FEC44F", "#FC8D59", "#D7301F", "#7F0000"),\n        N=256,\n    )\n    pColorAxis = figure.add_axes((0.800, 0.135, 0.145, 0.035))\n    pColorBar = mpl.colorbar.ColorbarBase(\n        pColorAxis,\n        cmap=pValueColorMap,\n        norm=Normalize(0.0, pColorMaximum),\n        orientation="horizontal",\n    )\n    pColorBar.set_ticks((0.0, 1.0, 2.0, 3.0))\n    pColorBar.set_label(r"$-\\log_{10}(p_{\\mathrm{FDR}})$", labelpad=2)\n    pColorBar.ax.xaxis.label.set_size(UNIFIED_FONT_SIZE)\n    pColorBar.ax.tick_params(length=3, pad=2)\n\n    # Enforce one font family and one font size for every text artist,\n    # including panel labels, color-bar labels, tick labels, and annotations.\n    for textArtist in figure.findobj(match=mpl.text.Text):\n        textArtist.set_fontfamily(FONT_FAMILY)\n        textArtist.set_fontsize(UNIFIED_FONT_SIZE)\n\n    return figure\n',
            'rebuild panel-g and panel-h by first extracting equal-sized brain tiles and only then integrating titles and color bars',
        )
        replaceRequired(
            "        for tileIndex, tileImage in enumerate(hcTiles):\n"
            "            tileLeft = groupLeft + tileIndex * (brainViewWidth + withinSystemGap)\n"
            "            addTileImageAxis(\n"
            "                figure,\n"
            "                tileImage,\n"
            "                tileLeft,\n"
            "                upperBottom,\n"
            "                brainViewWidth,\n"
            "                brainViewHeight,\n"
            "            )\n"
            "        for tileIndex, tileImage in enumerate(asdTiles):\n"
            "            tileLeft = groupLeft + tileIndex * (brainViewWidth + withinSystemGap)\n"
            "            addTileImageAxis(\n"
            "                figure,\n"
            "                tileImage,\n"
            "                tileLeft,\n"
            "                lowerBottom,\n"
            "                brainViewWidth,\n"
            "                brainViewHeight,\n"
            "            )\n",
            "        # Show complementary lateral views in the two vertical slots.\n"
            "        addTileImageAxis(\n"
            "            figure, hcTiles[0], groupLeft, upperBottom,\n"
            "            brainViewWidth, brainViewHeight,\n"
            "        )\n"
            "        addTileImageAxis(\n"
            "            figure, asdTiles[1], groupLeft, lowerBottom,\n"
            "            brainViewWidth, brainViewHeight,\n"
            "        )\n",
            "use the left lateral view above and the right lateral view below for each beta brain map",
        )
        # panel-h is now rebuilt from individual equal-sized brain tiles.
        replaceRequired(
            "    pColorMaximum = max(\n"
            "        3.0,\n"
            "        float(\n"
            "            np.nanmax(\n"
            "                [\n"
            "                    -np.log10(np.clip(parcelValues, MINIMUM_P_VALUE, 1.0))\n"
            "                    for parcelValues in pValueMaps.values()\n"
            "                ]\n"
            "            )\n"
            "        ),\n"
            "    )",
            "    pColorMaximum = 0.1",
            "set the FDR p-value color range",
        )
        replaceRequired(
            '        "negativeLog10FdrP",\n'
            '        ("#FFF7EC", "#FEC44F", "#FC8D59", "#D7301F", "#7F0000"),',
            '        "fdrPValue",\n'
            '        ("#2166AC", "#67A9CF", "#F7F7F7", "#EF8A62", "#B2182B"),',
            "change the rendered FDR p-value colormap",
        )
        replaceRequired(
            "        negativeLogPValues = -np.log10(\n"
            "            np.clip(parcelPValues, MINIMUM_P_VALUE, 1.0)\n"
            "        )\n"
            "        vertexLeft = mapParcelValuesToVertices(parcelLeft, negativeLogPValues)\n"
            "        vertexRight = mapParcelValuesToVertices(parcelRight, negativeLogPValues)",
            "        clippedPValues = np.clip(parcelPValues, 0.0, pColorMaximum)\n"
            "        vertexLeft = mapParcelValuesToVertices(parcelLeft, clippedPValues)\n"
            "        vertexRight = mapParcelValuesToVertices(parcelRight, clippedPValues)",
            "plot raw FDR-corrected p values",
        )
        replaceRequired(
            '        "negativeLog10FdrPFigure",\n'
            '        ("#FFF7EC", "#FEC44F", "#FC8D59", "#D7301F", "#7F0000"),',
            '        "fdrPValueFigure",\n'
            '        ("#2166AC", "#67A9CF", "#F7F7F7", "#EF8A62", "#B2182B"),',
            "change the figure FDR p-value colormap",
        )
        replaceRequired(
            "    pColorBar.set_ticks((0.0, 1.0, 2.0, 3.0))\n"
            '    pColorBar.set_label(r"$-\\log_{10}(p_{\\mathrm{FDR}})$", labelpad=2)',
            "    pColorBar.set_ticks((0.0, 0.05, 0.1))\n"
            '    pColorBar.set_ticklabels(("0", "0.05", "0.1"))\n'
            '    pColorBar.set_label(r"$p_{\\mathrm{FDR}}$", labelpad=2)',
            "label the raw FDR p-value color bar",
        )

        # The four row-4 heading texts share one y coordinate. Using va='top'
        # rather than va='center' aligns their upper edges exactly.
        for headingText in ("g", "h", "Mean Early SEC slope", "FDR-corrected p value"):
            replaceRequired(
                f'        "{headingText}",\n        ha="left",\n        va="center",',
                f'        "{headingText}",\n        ha="left",\n        va="top",',
                f"top-align the row-4 heading {headingText}",
            )

    # Enforce regular font weight for every text element except the panel
    # letters and the designated panel titles. This also normalizes ASD/HC,
    # L1-L4 labels, axis labels, tick labels, legends, annotations,
    # significance labels, and color-bar text.
    boldHeadingTextsByRow = {
        1: {
            "a",
            "b",
            "Group-averaged stepwise cortical propagation",
            "Propagation hierarchy across cortical levels (L1–L4)",
        },
        2: {
            "c",
            "d",
            "SEC-weighted step centroid",
            "Early propagation dynamics",
        },
        3: {
            "e",
            "f",
            "L4–L1 contrast\nEarly SEC slope",
            "L1 sensory\nEarly SEC slope",
            "L2 attention\nEarly SEC slope",
            "L3 control\nEarly SEC slope",
            "L4 DMN\nEarly SEC slope",
        },
        4: {
            "g",
            "h",
            "Mean Early SEC slope",
            "FDR-corrected p value",
        },
    }
    boldHeadingTexts = boldHeadingTextsByRow[rowNumber]
    saveFigureSignature = (
        "def saveFigure(figure: plt.Figure, outputStem: Path) -> None:\n"
    )
    saveFigureDoubleQuoteSignature = (
        "def saveFigure(figure: plt.Figure, outputStem: Path) -> None:\n"
    )
    normalizationCode = (
        "def normalizeFigureFontWeights(figure: plt.Figure) -> None:\n"
        f"    boldHeadingTexts = {boldHeadingTexts!r}\n"
        "    for textArtist in figure.findobj(match=mpl.text.Text):\n"
        f"        textArtist.set_fontfamily({FINAL_FONT_FAMILY!r})\n"
        f"        textArtist.set_fontsize({float(FINAL_FONT_SIZE):.1f})\n"
        "        textArtist.set_fontweight(\n"
        "            'heavy' if textArtist.get_text() in {'b', 'd'} else (\n"
        "                'bold' if textArtist.get_text() in boldHeadingTexts else 'normal'\n"
        "            )\n"
        "        )\n"
        "    for axis in figure.axes:\n"
        f"        axis.xaxis.label.set_fontfamily({FINAL_FONT_FAMILY!r})\n"
        f"        axis.yaxis.label.set_fontfamily({FINAL_FONT_FAMILY!r})\n"
        f"        axis.xaxis.label.set_fontsize({float(FINAL_FONT_SIZE):.1f})\n"
        f"        axis.yaxis.label.set_fontsize({float(FINAL_FONT_SIZE):.1f})\n"
        "        axis.xaxis.label.set_fontweight('normal')\n"
        "        axis.yaxis.label.set_fontweight('normal')\n"
        "\n"
        "def saveFigure(figure: plt.Figure, outputStem: Path) -> None:\n"
        "    normalizeFigureFontWeights(figure)\n"
        "    # Preserve all Matplotlib labels as editable SVG <text> elements.\n"
        "    plt.rcParams['svg.fonttype'] = 'none'\n"
    )
    if saveFigureSignature not in sourceText:
        raise RuntimeError(
            f"Could not apply embedded row {rowNumber} font-weight normalization."
        )
    sourceText = sourceText.replace(
        saveFigureSignature,
        normalizationCode,
        1,
    )

    rowModule = types.ModuleType(f"embedded_row_{rowNumber}")
    rowModule.__file__ = str(Path(__file__).resolve())
    rowModule.__dict__["__name__"] = f"embedded_row_{rowNumber}"
    exec(compile(sourceText, f"<embedded-row-{rowNumber}>", "exec"), rowModule.__dict__)
    return rowModule

def renderEmbeddedRows(scriptDirectory: Path) -> tuple[list[Path], list[Path]]:
    encodedRowSources = (ROW_1_SOURCE_B85, ROW_2_SOURCE_B85, ROW_3_SOURCE_B85, ROW_4_SOURCE_B85)
    rowImagePaths: list[Path] = []
    rowSvgPaths: list[Path] = []
    for rowNumber, encodedSource in enumerate(encodedRowSources, start=1):
        rowModule = loadEmbeddedRowModule(encodedSource, rowNumber)
        rowModule.main()
        rowOutputDirectory = scriptDirectory / f"abide1-propagation-hierarchy-row-{rowNumber}"
        rowImagePath = rowOutputDirectory / f"abide1-propagation-hierarchy-row-{rowNumber}.png"
        rowSvgPath = rowOutputDirectory / f"abide1-propagation-hierarchy-row-{rowNumber}.svg"
        if not rowImagePath.exists():
            raise FileNotFoundError(f"Embedded row {rowNumber} did not create its PNG output.")
        if not rowSvgPath.exists():
            raise FileNotFoundError(f"Embedded row {rowNumber} did not create its SVG output.")
        rowImagePaths.append(rowImagePath)
        rowSvgPaths.append(rowSvgPath)
    return rowImagePaths, rowSvgPaths


def validateCompositeLayout() -> None:
    """Fail early if a source crop or destination slot is malformed."""
    validAlignments = {"left", "center", "right"}
    validVerticalAlignments = {"top", "center", "bottom"}
    seenNames: set[str] = set()

    for tile in COMPOSITE_LAYOUT_TILES:
        if tile.name in seenNames:
            raise ValueError(f"Duplicate layout tile name: {tile.name}.")
        seenNames.add(tile.name)

        if not 0 <= tile.row_index < len(ROW_HEIGHT_PIXELS):
            raise ValueError(f"Invalid row index for {tile.name}: {tile.row_index}.")
        if tile.source.width <= 0 or tile.source.height <= 0:
            raise ValueError(f"Invalid source rectangle for {tile.name}: {tile.source}.")
        if tile.slot.width <= 0 or tile.slot.height <= 0:
            raise ValueError(f"Invalid destination slot for {tile.name}: {tile.slot}.")
        if tile.horizontal_alignment not in validAlignments:
            raise ValueError(
                f"Invalid horizontal alignment for {tile.name}: "
                f"{tile.horizontal_alignment}."
            )
        if tile.vertical_alignment not in validVerticalAlignments:
            raise ValueError(
                f"Invalid vertical alignment for {tile.name}: "
                f"{tile.vertical_alignment}."
            )

        rowHeight = ROW_HEIGHT_PIXELS[tile.row_index]
        if (
            tile.source.left < 0
            or tile.source.top < 0
            or tile.source.right > ROW_SOURCE_WIDTH_PIXELS
            or tile.source.bottom > rowHeight
        ):
            raise ValueError(
                f"Source rectangle for {tile.name} exceeds row "
                f"{tile.row_index + 1}: {tile.source}."
            )
        if (
            tile.slot.left < 0
            or tile.slot.top < 0
            or tile.slot.right > FIGURE_WIDTH_PIXELS
            or tile.slot.bottom > FIGURE_HEIGHT_PIXELS
        ):
            raise ValueError(
                f"Destination slot for {tile.name} exceeds the final canvas: "
                f"{tile.slot}."
            )


TEXT_BEARING_TILE_NAMES = {
    "top-a",
    "top-b",
    "middle-b",
    "middle-c",
    "middle-d",
    "bottom-f-plot",
    "bottom-g-plot",
    "bottom-h-plot",
}

# All cortical brain-map tiles are normalized from their actual visible
# (non-white) brain bounding boxes.  This removes source-specific white
# margins and makes panel e/f/g/h use the same visible brain size and the
# same HC-ASD gap.
BRAIN_TILE_NAMES = {
    "middle-hc",
    "middle-asd",
    "bottom-f-hc",
    "bottom-f-asd",
    "bottom-g-hc",
    "bottom-g-asd",
    "bottom-h-hc",
    "bottom-h-asd",
}


def validateTextTileScale() -> None:
    """Ensure all tiles containing typography are assembled at exactly 1:1 scale."""
    for tileName in TEXT_BEARING_TILE_NAMES:
        tile = LAYOUT_TILE_BY_NAME[tileName]
        if (
            abs(tile.slot.width - tile.source.width) > 1e-9
            or abs(tile.slot.height - tile.source.height) > 1e-9
        ):
            raise ValueError(
                f"Text-bearing tile {tileName} must be assembled at 1:1 scale: "
                f"source={tile.source.width}x{tile.source.height}, "
                f"slot={tile.slot.width}x{tile.slot.height}."
            )


def fitTileIntoSlot(tile: LayoutTile) -> PixelRect:
    """Contain a tile inside its slot while preserving source aspect ratio."""
    scale = min(
        tile.slot.width / tile.source.width,
        tile.slot.height / tile.source.height,
    )
    fittedWidth = tile.source.width * scale
    fittedHeight = tile.source.height * scale

    if tile.horizontal_alignment == "left":
        fittedLeft = tile.slot.left
    elif tile.horizontal_alignment == "right":
        fittedLeft = tile.slot.right - fittedWidth
    else:
        fittedLeft = tile.slot.left + 0.5 * (tile.slot.width - fittedWidth)

    if tile.vertical_alignment == "top":
        fittedTop = tile.slot.top
    elif tile.vertical_alignment == "bottom":
        fittedTop = tile.slot.bottom - fittedHeight
    else:
        fittedTop = tile.slot.top + 0.5 * (tile.slot.height - fittedHeight)

    return PixelRect(
        fittedLeft,
        fittedTop,
        fittedLeft + fittedWidth,
        fittedTop + fittedHeight,
    )


def createVisibleTileMask(tileImage: Image.Image) -> Image.Image:
    """Create a soft-safe paste mask that removes only the white background."""
    whiteBackground = Image.new("RGB", tileImage.size, "white")
    visibleMask = ImageChops.difference(tileImage, whiteBackground).convert("L")
    return visibleMask.point(
        lambda pixelValue: 255 if pixelValue > VISIBLE_MASK_THRESHOLD else 0
    )


def getVisibleContentBounds(tileImage: Image.Image) -> tuple[int, int, int, int]:
    """Return the non-white content bounds of one brain-map source tile."""
    whiteBackground = Image.new("RGB", tileImage.size, "white")
    differenceImage = ImageChops.difference(tileImage, whiteBackground).convert("L")
    visibleMask = differenceImage.point(
        lambda pixelValue: 255 if pixelValue > VISIBLE_MASK_THRESHOLD else 0
    )
    visibleBounds = visibleMask.getbbox()
    if visibleBounds is None:
        raise RuntimeError("A brain-map tile contains no visible cortical image.")
    return visibleBounds


def getBrainTemplateVisibleSize(rowImages: list[Image.Image]) -> tuple[int, int]:
    """Use panel-e HC as the pixel template for every e/f/g/h brain map."""
    templateTile = LAYOUT_TILE_BY_NAME["middle-hc"]
    templateSource = rowImages[templateTile.row_index].crop(
        templateTile.source.as_crop_box()
    )
    left, top, right, bottom = getVisibleContentBounds(templateSource)
    visibleWidth = right - left
    visibleHeight = bottom - top

    # Reproduce panel e's existing contain-scale before removing its white
    # margins.  The resulting visible size therefore keeps panel e unchanged
    # in scale while forcing f/g/h to match it exactly.
    fittedRect = fitTileIntoSlot(templateTile)
    sourceScale = min(
        fittedRect.width / templateSource.width,
        fittedRect.height / templateSource.height,
    )
    targetWidth = max(1, round(visibleWidth * sourceScale))
    targetHeight = max(1, round(visibleHeight * sourceScale))
    return targetWidth, targetHeight


def getBrainReferencePanelName(brainTileName: str) -> str:
    """Return the final panel name associated with one brain tile."""
    if brainTileName.startswith("middle-"):
        return "e"
    if brainTileName.startswith("bottom-f-"):
        return "f"
    if brainTileName.startswith("bottom-g-"):
        return "g"
    if brainTileName.startswith("bottom-h-"):
        return "h"
    raise KeyError(f"Unknown brain tile name: {brainTileName}")


# These are the actual right edges of the visible statistical axes in the
# final 1800 px composite, not the right edges of the larger text-bearing
# crop tiles.  Using the true axis edges removes the unequal hidden white
# margins that previously made e/f/g/h look different.
VISIBLE_STATISTICAL_PLOT_RIGHT_PIXELS = {
    "e": 1543.0,
    "f": 322.0,
    "g": 904.0,
    "h": 1543.0,
}

# Every visible cortical brain begins exactly this many pixels to the right
# of the visible box/violin plot axis.
UNIFIED_VISIBLE_BRAIN_PLOT_GAP_PIXELS = 30.0


def renderNormalizedBrainTile(
    rowImages: list[Image.Image],
    tile: LayoutTile,
    targetVisibleSize: tuple[int, int],
) -> Image.Image:
    """Render the exact normalized brain tile used by both PNG and SVG."""
    sourceTile = rowImages[tile.row_index].crop(tile.source.as_crop_box())
    visibleBounds = getVisibleContentBounds(sourceTile)
    visibleBrain = sourceTile.crop(visibleBounds)
    targetWidth, targetHeight = targetVisibleSize
    return visibleBrain.resize(
        (targetWidth, targetHeight),
        Image.Resampling.LANCZOS,
    )


def getNormalizedBrainPlacement(
    tile: LayoutTile,
    targetVisibleSize: tuple[int, int],
) -> tuple[int, int]:
    """Return the exact final-canvas position shared by PNG and SVG brains."""
    targetWidth, targetHeight = targetVisibleSize
    panelName = getBrainReferencePanelName(tile.name)
    visiblePlotRight = VISIBLE_STATISTICAL_PLOT_RIGHT_PIXELS[panelName]
    pasteLeft = round(visiblePlotRight + UNIFIED_VISIBLE_BRAIN_PLOT_GAP_PIXELS)
    pasteTop = round(tile.slot.center_y - targetHeight / 2.0)

    if pasteLeft + targetWidth > FIGURE_WIDTH_PIXELS:
        raise ValueError(
            f"Brain tile {tile.name} exceeds the final canvas after applying "
            f"the unified visible plot gap: right={pasteLeft + targetWidth}px."
        )
    return pasteLeft, pasteTop


def pasteNormalizedBrainTile(
    completeImage: Image.Image,
    rowImages: list[Image.Image],
    tile: LayoutTile,
    targetVisibleSize: tuple[int, int],
) -> None:
    """Paste one brain using the same pixels later embedded in the SVG."""
    resizedBrain = renderNormalizedBrainTile(
        rowImages,
        tile,
        targetVisibleSize,
    )
    pasteLeft, pasteTop = getNormalizedBrainPlacement(tile, targetVisibleSize)
    completeImage.paste(
        resizedBrain,
        (pasteLeft, pasteTop),
        createVisibleTileMask(resizedBrain),
    )


def brainTileToTransparentPngDataUri(brainImage: Image.Image) -> str:
    """Encode a normalized brain tile with the PNG paste mask as alpha."""
    alphaMask = createVisibleTileMask(brainImage)
    rgbaImage = Image.new("RGBA", brainImage.size, (255, 255, 255, 0))
    rgbaImage.paste(brainImage, (0, 0), alphaMask)
    buffer = io.BytesIO()
    rgbaImage.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def pasteLayoutTile(
    completeImage: Image.Image,
    rowImages: list[Image.Image],
    tile: LayoutTile,
) -> None:
    """Crop, fit, resize, and paste one layout tile using the shared rules."""
    fittedRect = fitTileIntoSlot(tile)
    fittedWidth = max(1, round(fittedRect.width))
    fittedHeight = max(1, round(fittedRect.height))
    sourceTile = rowImages[tile.row_index].crop(tile.source.as_crop_box())
    resizedTile = sourceTile.resize(
        (fittedWidth, fittedHeight),
        Image.Resampling.LANCZOS,
    )
    completeImage.paste(
        resizedTile,
        (round(fittedRect.left), round(fittedRect.top)),
        createVisibleTileMask(resizedTile),
    )


def assembleCompleteFigure(rowImagePaths: list[Path], outputPath: Path) -> None:
    validateCompositeLayout()
    validateTextTileScale()
    completeImage = Image.new(
        "RGB",
        (FIGURE_WIDTH_PIXELS, FIGURE_HEIGHT_PIXELS),
        "white",
    )
    rowImages = [
        Image.open(rowImagePath).convert("RGB")
        for rowImagePath in rowImagePaths
    ]
    for rowImage, rowImagePath, expectedHeight in zip(
        rowImages,
        rowImagePaths,
        ROW_HEIGHT_PIXELS,
    ):
        if (
            rowImage.height != expectedHeight
            or abs(rowImage.width - ROW_SOURCE_WIDTH_PIXELS) > 1
        ):
            raise ValueError(
                f"Unexpected row size for {rowImagePath.name}: {rowImage.size}."
            )

    brainTemplateVisibleSize = getBrainTemplateVisibleSize(rowImages)
    for tile in COMPOSITE_LAYOUT_TILES:
        if tile.name in BRAIN_TILE_NAMES:
            pasteNormalizedBrainTile(
                completeImage,
                rowImages,
                tile,
                brainTemplateVisibleSize,
            )
        else:
            pasteLayoutTile(completeImage, rowImages, tile)

    annotationDrawer = ImageDraw.Draw(completeImage)

    # Final raster annotations must use the same Arial 25 px typography as
    # the Matplotlib-generated row figures. Search standard OS locations and
    # fail explicitly rather than silently substituting another font family.
    arialRegularCandidates = (
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"),
        Path("/usr/share/fonts/truetype/msttcorefonts/arial.ttf"),
    )
    arialBoldCandidates = (
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("/Library/Fonts/Arial Bold.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf"),
        Path("/usr/share/fonts/truetype/msttcorefonts/arialbd.ttf"),
    )

    regularFontPath = next((path for path in arialRegularCandidates if path.exists()), None)
    boldFontPath = next((path for path in arialBoldCandidates if path.exists()), None)
    if regularFontPath is None or boldFontPath is None:
        raise FileNotFoundError(
            "Arial regular/bold font files are required to keep every label in Arial. "
            "Install Arial on this system before rendering the final figure."
        )

    arialFont = ImageFont.truetype(str(regularFontPath), size=FINAL_FONT_SIZE)
    arialBoldFont = ImageFont.truetype(str(boldFontPath), size=FINAL_FONT_SIZE)
    for labelLeft, labelTop, labelText in getBrainGroupLabelPositions():
        annotationDrawer.text(
            (labelLeft, labelTop),
            labelText,
            fill="black",
            font=arialFont,
            anchor="lm",
        )
    for labelLeft, labelTop, labelText in BOTTOM_PANEL_LABEL_POSITIONS:
        annotationDrawer.text(
            (labelLeft, labelTop),
            labelText,
            fill="black",
            font=arialBoldFont,
            anchor="lt",
        )
    completeImage.save(outputPath, dpi=(600, 600))

    # The publication canvas width is fixed and must never be altered by
    # cropping, tight bounding boxes, or downstream layout changes.
    with Image.open(outputPath) as savedImage:
        if savedImage.width != FIGURE_WIDTH_PIXELS:
            raise RuntimeError(
                f"Final figure width changed unexpectedly: {savedImage.width}px; "
                f"expected exactly {FIGURE_WIDTH_PIXELS}px."
            )


SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)


def localSvgTag(tag: str) -> str:
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def parseSvgViewBox(root: ET.Element, svgPath: Path) -> tuple[float, float, float, float]:
    viewBoxText = root.get("viewBox")
    if viewBoxText is None:
        raise ValueError(f"SVG viewBox is missing in {svgPath.name}.")
    viewBoxValues = tuple(float(value) for value in viewBoxText.replace(",", " ").split())
    if len(viewBoxValues) != 4:
        raise ValueError(f"Unexpected SVG viewBox in {svgPath.name}.")
    return viewBoxValues


def prefixSvgTreeIds(root: ET.Element, elementPrefix: str) -> None:
    prefixedIds: dict[str, str] = {}
    for element in root.iter():
        elementId = element.get("id")
        if elementId:
            prefixedIds[elementId] = f"{elementPrefix}-{elementId}"
    if not prefixedIds:
        return

    urlReferencePattern = re.compile(r"url\(#([^)]+)\)")

    for element in root.iter():
        elementId = element.get("id")
        if elementId in prefixedIds:
            element.set("id", prefixedIds[elementId])
        for attributeName, attributeValue in list(element.attrib.items()):
            if not isinstance(attributeValue, str):
                continue
            updatedValue = urlReferencePattern.sub(
                lambda match: f"url(#{prefixedIds.get(match.group(1), match.group(1))})",
                attributeValue,
            )
            if updatedValue.startswith("#") and updatedValue[1:] in prefixedIds:
                updatedValue = f"#{prefixedIds[updatedValue[1:]]}"
            elif updatedValue in prefixedIds:
                updatedValue = prefixedIds[updatedValue]
            element.set(attributeName, updatedValue)


def removeRootPatch(root: ET.Element) -> None:
    """Remove the Matplotlib figure-wide white background patch recursively.

    Matplotlib places ``patch_1`` inside ``figure_1`` rather than directly
    under the root <svg>.  Keeping that patch in a tile whose composite crop
    is no longer implemented by clipPath creates a huge white rectangle that
    covers previously assembled panels.  The final 1800 x 1320 SVG already
    owns one white background rectangle, so every row-level patch_1 must be
    removed before panel extraction.
    """
    for parent in list(root.iter()):
        for child in list(parent):
            childId = child.get("id", "")
            if childId == "patch_1" or childId.endswith("-patch_1"):
                parent.remove(child)


def identityTransform() -> tuple[float, float, float, float, float, float]:
    return (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def multiplyTransforms(
    left: tuple[float, float, float, float, float, float],
    right: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float, float, float]:
    a1, b1, c1, d1, e1, f1 = left
    a2, b2, c2, d2, e2, f2 = right
    return (
        a1 * a2 + c1 * b2,
        b1 * a2 + d1 * b2,
        a1 * c2 + c1 * d2,
        b1 * c2 + d1 * d2,
        a1 * e2 + c1 * f2 + e1,
        b1 * e2 + d1 * f2 + f1,
    )


def applyTransform(
    transform: tuple[float, float, float, float, float, float],
    xValue: float,
    yValue: float,
) -> tuple[float, float]:
    aValue, bValue, cValue, dValue, eValue, fValue = transform
    return (
        aValue * xValue + cValue * yValue + eValue,
        bValue * xValue + dValue * yValue + fValue,
    )


def parseTransformAttribute(
    transformText: str | None,
) -> tuple[float, float, float, float, float, float]:
    if not transformText:
        return identityTransform()
    transform = identityTransform()
    for functionName, argumentText in re.findall(r"([A-Za-z]+)\s*\(([^)]*)\)", transformText):
        rawArguments = [
            value for value in re.split(r"[\s,]+", argumentText.strip()) if value
        ]
        argumentValues = [float(value) for value in rawArguments]
        if functionName == "translate":
            txValue = argumentValues[0] if argumentValues else 0.0
            tyValue = argumentValues[1] if len(argumentValues) > 1 else 0.0
            component = (1.0, 0.0, 0.0, 1.0, txValue, tyValue)
        elif functionName == "scale":
            sxValue = argumentValues[0] if argumentValues else 1.0
            syValue = argumentValues[1] if len(argumentValues) > 1 else sxValue
            component = (sxValue, 0.0, 0.0, syValue, 0.0, 0.0)
        elif functionName == "matrix":
            if len(argumentValues) != 6:
                continue
            component = tuple(argumentValues)
        elif functionName == "rotate":
            if not argumentValues:
                continue
            angleRadians = math.radians(argumentValues[0])
            cosineValue = math.cos(angleRadians)
            sineValue = math.sin(angleRadians)
            rotation = (
                cosineValue, sineValue,
                -sineValue, cosineValue,
                0.0, 0.0,
            )
            if len(argumentValues) >= 3:
                centerX = argumentValues[1]
                centerY = argumentValues[2]
                component = multiplyTransforms(
                    multiplyTransforms(
                        (1.0, 0.0, 0.0, 1.0, centerX, centerY),
                        rotation,
                    ),
                    (1.0, 0.0, 0.0, 1.0, -centerX, -centerY),
                )
            else:
                component = rotation
        else:
            # Unknown transforms are rare in Matplotlib SVG. Retaining the
            # element is safer than deleting it from an uncertain position.
            return identityTransform()
        transform = multiplyTransforms(transform, component)
    return transform


def transformBoundingBox(
    bounds: tuple[float, float, float, float],
    transform: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float]:
    left, top, right, bottom = bounds
    cornerPoints = (
        applyTransform(transform, left, top),
        applyTransform(transform, right, top),
        applyTransform(transform, left, bottom),
        applyTransform(transform, right, bottom),
    )
    xs = [point[0] for point in cornerPoints]
    ys = [point[1] for point in cornerPoints]
    return (min(xs), min(ys), max(xs), max(ys))


def unionBoundingBoxes(
    boxes: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float] | None:
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def elementIntersectsCrop(
    bounds: tuple[float, float, float, float] | None,
    cropBounds: tuple[float, float, float, float],
) -> bool:
    if bounds is None:
        return True
    left, top, right, bottom = bounds
    cropLeft, cropTop, cropRight, cropBottom = cropBounds
    return not (
        right < cropLeft
        or left > cropRight
        or bottom < cropTop
        or top > cropBottom
    )


def parsePointList(pointsText: str | None) -> list[tuple[float, float]]:
    if not pointsText:
        return []
    numberValues = [float(value) for value in re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", pointsText)]
    return [
        (numberValues[index], numberValues[index + 1])
        for index in range(0, len(numberValues) - 1, 2)
    ]


def parseSvgStyle(styleText: str | None) -> dict[str, str]:
    if not styleText:
        return {}
    styleValues: dict[str, str] = {}
    for declaration in styleText.split(";"):
        if ":" not in declaration:
            continue
        key, value = declaration.split(":", 1)
        styleValues[key.strip()] = value.strip()
    return styleValues


def getSvgPresentationValue(
    element: ET.Element,
    attributeName: str,
    defaultValue: str | None = None,
) -> str | None:
    directValue = element.get(attributeName)
    if directValue is not None:
        return directValue
    return parseSvgStyle(element.get("style")).get(attributeName, defaultValue)


def approximateTextBounds(
    element: ET.Element,
    transform: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float] | None:
    # Matplotlib normally emits x/y explicitly. If one is absent, zero is
    # correct for text positioned solely through a translate transform.
    try:
        xValue = float(str(element.get("x", "0")).replace("px", ""))
        yValue = float(str(element.get("y", "0")).replace("px", ""))
    except ValueError:
        return None

    fontSizeText = getSvgPresentationValue(
        element, "font-size", str(FINAL_FONT_SIZE)
    ) or str(FINAL_FONT_SIZE)
    fontSizeMatch = re.search(
        r"[-+]?(?:\d+\.\d*|\.\d+|\d+)", fontSizeText
    )
    fontSize = (
        float(fontSizeMatch.group(0))
        if fontSizeMatch
        else float(FINAL_FONT_SIZE)
    )
    textContent = "".join(element.itertext()) or "W"
    # The bounding box only decides whether a complete text object is outside
    # a tile; a slight overestimate is intentional so edge labels are retained.
    textWidth = max(fontSize * 0.60, len(textContent) * fontSize * 0.72)
    textHeight = fontSize * 1.35
    anchor = getSvgPresentationValue(element, "text-anchor", "start") or "start"
    if anchor == "middle":
        left = xValue - 0.5 * textWidth
    elif anchor == "end":
        left = xValue - textWidth
    else:
        left = xValue

    baseline = (
        getSvgPresentationValue(element, "dominant-baseline", "alphabetic")
        or "alphabetic"
    )
    if baseline in {"middle", "central"}:
        top = yValue - 0.5 * textHeight
    elif baseline in {"hanging", "text-before-edge"}:
        top = yValue
    else:
        top = yValue - 0.85 * textHeight
    return transformBoundingBox(
        (left, top, left + textWidth, top + textHeight),
        transform,
    )


def approximatePathBounds(
    pathData: str | None,
    transform: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float] | None:
    if not pathData:
        return None
    numberValues = [float(value) for value in re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", pathData)]
    if len(numberValues) < 2:
        return None
    points = [
        (numberValues[index], numberValues[index + 1])
        for index in range(0, len(numberValues) - 1, 2)
    ]
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return transformBoundingBox((min(xs), min(ys), max(xs), max(ys)), transform)


def approximateElementBounds(
    element: ET.Element,
    inheritedTransform: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float] | None:
    currentTransform = multiplyTransforms(
        inheritedTransform,
        parseTransformAttribute(element.get("transform")),
    )
    tagName = localSvgTag(element.tag)

    if tagName in {"g", "svg", "a"}:
        childBounds = [
            bounds
            for child in list(element)
            for bounds in [approximateElementBounds(child, currentTransform)]
            if bounds is not None and localSvgTag(child.tag) != "defs"
        ]
        return unionBoundingBoxes(childBounds)
    if tagName == "defs":
        return None
    if tagName == "path":
        return approximatePathBounds(element.get("d"), currentTransform)
    if tagName == "rect":
        xValue = float(element.get("x", "0"))
        yValue = float(element.get("y", "0"))
        widthValue = float(element.get("width", "0"))
        heightValue = float(element.get("height", "0"))
        return transformBoundingBox((xValue, yValue, xValue + widthValue, yValue + heightValue), currentTransform)
    if tagName == "line":
        x1Value = float(element.get("x1", "0"))
        y1Value = float(element.get("y1", "0"))
        x2Value = float(element.get("x2", "0"))
        y2Value = float(element.get("y2", "0"))
        return transformBoundingBox((min(x1Value, x2Value), min(y1Value, y2Value), max(x1Value, x2Value), max(y1Value, y2Value)), currentTransform)
    if tagName in {"polyline", "polygon"}:
        points = parsePointList(element.get("points"))
        if not points:
            return None
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return transformBoundingBox((min(xs), min(ys), max(xs), max(ys)), currentTransform)
    if tagName == "circle":
        cxValue = float(element.get("cx", "0"))
        cyValue = float(element.get("cy", "0"))
        radius = float(element.get("r", "0"))
        return transformBoundingBox((cxValue - radius, cyValue - radius, cxValue + radius, cyValue + radius), currentTransform)
    if tagName == "ellipse":
        cxValue = float(element.get("cx", "0"))
        cyValue = float(element.get("cy", "0"))
        rxValue = float(element.get("rx", "0"))
        ryValue = float(element.get("ry", "0"))
        return transformBoundingBox((cxValue - rxValue, cyValue - ryValue, cxValue + rxValue, cyValue + ryValue), currentTransform)
    if tagName == "text":
        return approximateTextBounds(element, currentTransform)
    if tagName in {"image", "use"}:
        xValue = float(element.get("x", "0"))
        yValue = float(element.get("y", "0"))
        widthValue = float(element.get("width", "0"))
        heightValue = float(element.get("height", "0"))
        return transformBoundingBox((xValue, yValue, xValue + widthValue, yValue + heightValue), currentTransform)
    return None


def cloneElementWithoutChildren(element: ET.Element) -> ET.Element:
    cloned = ET.Element(element.tag, dict(element.attrib))
    cloned.text = element.text
    cloned.tail = element.tail
    return cloned


def boundingBoxCenterIsInsideCrop(
    bounds: tuple[float, float, float, float] | None,
    cropBounds: tuple[float, float, float, float],
    tolerance: float = 0.5,
) -> bool:
    if bounds is None:
        return False
    centerX = 0.5 * (bounds[0] + bounds[2])
    centerY = 0.5 * (bounds[1] + bounds[3])
    return (
        cropBounds[0] - tolerance <= centerX <= cropBounds[2] + tolerance
        and cropBounds[1] - tolerance <= centerY <= cropBounds[3] + tolerance
    )


def isMatplotlibAxesGroup(element: ET.Element) -> bool:
    elementId = element.get("id", "")
    return (
        localSvgTag(element.tag) == "g"
        and re.search(r"(?:^|-)axes_\d+$", elementId) is not None
    )


def getAxesBackgroundBounds(
    axesElement: ET.Element,
    inheritedTransform: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float] | None:
    """Return the rectangular plotting-area patch of a Matplotlib axes group.

    The first direct ``patch_N`` child of an ordinary Matplotlib axes is the
    white axes background.  Its center is a much safer panel-membership test
    than recursively estimating the bounds of violin/scatter paths.
    """
    axesTransform = multiplyTransforms(
        inheritedTransform,
        parseTransformAttribute(axesElement.get("transform")),
    )
    for child in list(axesElement):
        childId = child.get("id", "")
        if not re.search(r"(?:^|-)patch_\d+$", childId):
            continue
        candidateBounds: list[tuple[float, float, float, float]] = []
        for descendant in child.iter():
            if localSvgTag(descendant.tag) not in {"path", "rect"}:
                continue
            bounds = approximateElementBounds(descendant, axesTransform)
            if bounds is not None:
                candidateBounds.append(bounds)
        patchBounds = unionBoundingBoxes(candidateBounds)
        if patchBounds is None:
            continue
        if (patchBounds[2] - patchBounds[0]) > 10 and (patchBounds[3] - patchBounds[1]) > 10:
            return patchBounds
    return None


def pruneSvgElementToCrop(
    element: ET.Element,
    cropBounds: tuple[float, float, float, float],
    inheritedTransform: tuple[float, float, float, float, float, float],
) -> ET.Element | None:
    """Extract one panel without creating a new composite clipping mask.

    Ordinary Matplotlib axes are treated atomically: if the center of their
    background patch belongs to the requested source crop, the complete axes
    group is retained.  This is critical for raincloud/violin plots because
    their FillBetween/PolyCollection paths are complex Bézier geometry and
    must not be partially deleted by an approximate SVG-path bounding parser.

    Axes belonging to neighbouring panels are deleted as whole objects.  For
    figure-level labels, headings and annotations, the object's center must be
    inside the crop; this prevents neighbouring scientific-notation labels and
    titles from leaking into the extracted panel.
    """
    tagName = localSvgTag(element.tag)
    currentTransform = multiplyTransforms(
        inheritedTransform,
        parseTransformAttribute(element.get("transform")),
    )

    if tagName in {"metadata", "style"}:
        return copy.deepcopy(element)

    if tagName == "defs":
        clonedDefs = cloneElementWithoutChildren(element)
        for child in list(element):
            clonedDefs.append(copy.deepcopy(child))
        return clonedDefs if list(clonedDefs) else None

    if isMatplotlibAxesGroup(element):
        axesBounds = getAxesBackgroundBounds(element, inheritedTransform)
        if axesBounds is not None:
            if boundingBoxCenterIsInsideCrop(axesBounds, cropBounds):
                return copy.deepcopy(element)
            return None
        # Container/helper axes without an ordinary background patch are
        # processed recursively so their figure-level labels can still survive.

    if tagName in {"g", "svg", "a"}:
        clonedContainer = cloneElementWithoutChildren(element)
        for child in list(element):
            prunedChild = pruneSvgElementToCrop(child, cropBounds, currentTransform)
            if prunedChild is not None:
                clonedContainer.append(prunedChild)
        if list(clonedContainer):
            return clonedContainer
        return None

    bounds = approximateElementBounds(element, inheritedTransform)
    if bounds is None:
        # Unknown structural nodes are retained conservatively; unused defs are
        # removed later after all references are known.
        return copy.deepcopy(element)

    if not boundingBoxCenterIsInsideCrop(bounds, cropBounds):
        return None
    return copy.deepcopy(element)

def collectReferencedDefinitionIds(root: ET.Element) -> set[str]:
    referencedIds: set[str] = set()
    urlReferencePattern = re.compile(r"url\(#([^)]+)\)")
    for element in root.iter():
        for attributeValue in element.attrib.values():
            if not isinstance(attributeValue, str):
                continue
            referencedIds.update(urlReferencePattern.findall(attributeValue))
            if attributeValue.startswith("#"):
                referencedIds.add(attributeValue[1:])
    return referencedIds


def boundingBoxIsInside(
    innerBounds: tuple[float, float, float, float],
    outerBounds: tuple[float, float, float, float],
    tolerance: float = 0.05,
) -> bool:
    return (
        innerBounds[0] >= outerBounds[0] - tolerance
        and innerBounds[1] >= outerBounds[1] - tolerance
        and innerBounds[2] <= outerBounds[2] + tolerance
        and innerBounds[3] <= outerBounds[3] + tolerance
    )


def getClipPathRectangleBounds(root: ET.Element) -> dict[str, tuple[float, float, float, float]]:
    """Read ordinary Matplotlib rectangular axis clip paths."""
    clipBoundsById: dict[str, tuple[float, float, float, float]] = {}
    for element in root.iter():
        if localSvgTag(element.tag) != "clipPath":
            continue
        clipId = element.get("id")
        if not clipId:
            continue
        clipTransform = parseTransformAttribute(element.get("transform"))
        childBounds: list[tuple[float, float, float, float]] = []
        for child in list(element):
            if localSvgTag(child.tag) != "rect":
                # Keep nonrectangular clips untouched; they may be semantically necessary.
                childBounds = []
                break
            bounds = approximateElementBounds(child, clipTransform)
            if bounds is not None:
                childBounds.append(bounds)
        unionBounds = unionBoundingBoxes(childBounds)
        if unionBounds is not None:
            clipBoundsById[clipId] = unionBounds
    return clipBoundsById


def removeClipPathFromElement(element: ET.Element) -> None:
    clipAttribute = element.get("clip-path")
    if clipAttribute is not None:
        element.attrib.pop("clip-path", None)

    styleValues = parseSvgStyle(element.get("style"))
    if "clip-path" in styleValues:
        styleValues.pop("clip-path", None)
        if styleValues:
            element.set(
                "style",
                "; ".join(f"{key}: {value}" for key, value in styleValues.items()),
            )
        else:
            element.attrib.pop("style", None)


def getElementClipPathId(element: ET.Element) -> str | None:
    clipValue = element.get("clip-path")
    if clipValue is None:
        clipValue = parseSvgStyle(element.get("style")).get("clip-path")
    if not clipValue:
        return None
    match = re.fullmatch(r"url\(#([^)]+)\)", clipValue.strip())
    return match.group(1) if match else None


def svgPathUsesOnlyStraightSegments(pathData: str | None) -> bool:
    if not pathData:
        return False
    commandLetters = set(re.findall(r"[A-Za-z]", pathData))
    return commandLetters.issubset(set("MmLlHhVvZz"))


def removeProvablyUnneededMatplotlibClipPaths(root: ET.Element) -> int:
    """Remove only clip paths whose removal is geometrically low risk.

    Violin/FillBetween/PolyCollection objects, rasterized scatter images and
    curved paths keep their original Matplotlib axis clip.  Simple straight
    lines, rectangles, text and uses lose the clip only when their complete
    transformed bounding box is already inside the rectangular clip region.
    This keeps Illustrator masks to the cases that can actually affect the
    rendered panel while avoiding visual changes.
    """
    clipBoundsById = getClipPathRectangleBounds(root)
    removedCount = 0

    def elementIsSafeForClipRemoval(element: ET.Element) -> bool:
        tagName = localSvgTag(element.tag)
        elementId = element.get("id", "")
        riskyIdTokens = (
            "Collection",
            "PathCollection",
            "PolyCollection",
            "FillBetween",
        )
        if any(token in elementId for token in riskyIdTokens):
            return False
        if tagName == "image":
            return False
        if tagName == "path":
            return svgPathUsesOnlyStraightSegments(element.get("d"))
        return tagName in {"line", "rect", "circle", "ellipse", "polyline", "polygon", "text", "use"}

    def visit(
        element: ET.Element,
        inheritedTransform: tuple[float, float, float, float, float, float],
    ) -> None:
        nonlocal removedCount
        if localSvgTag(element.tag) == "defs":
            return

        clipId = getElementClipPathId(element)
        if clipId in clipBoundsById and elementIsSafeForClipRemoval(element):
            elementBounds = approximateElementBounds(element, inheritedTransform)
            if (
                elementBounds is not None
                and boundingBoxIsInside(elementBounds, clipBoundsById[clipId])
            ):
                removeClipPathFromElement(element)
                removedCount += 1

        currentTransform = multiplyTransforms(
            inheritedTransform,
            parseTransformAttribute(element.get("transform")),
        )
        for child in list(element):
            visit(child, currentTransform)

    for child in list(root):
        visit(child, identityTransform())
    return removedCount

def pruneUnreferencedDefs(root: ET.Element) -> None:
    referencedIds = collectReferencedDefinitionIds(root)
    for parent in root.iter():
        for child in list(parent):
            if localSvgTag(child.tag) != "defs":
                continue
            for defChild in list(child):
                defId = defChild.get("id")
                if defId is None:
                    continue
                if localSvgTag(defChild.tag) == "clipPath" and defId not in referencedIds:
                    child.remove(defChild)
                elif defId not in referencedIds and localSvgTag(defChild.tag) in {"path", "marker", "pattern", "symbol"}:
                    child.remove(defChild)
            if not list(child):
                parent.remove(child)


def elementTreeChildrenToString(element: ET.Element) -> str:
    return "".join(
        ET.tostring(child, encoding="unicode")
        for child in list(element)
    )


def prepareSvgTileContent(
    svgPath: Path,
    elementPrefix: str,
    cropBoundsPixels: tuple[float, float, float, float],
    sourcePixelSize: tuple[float, float],
) -> tuple[str, tuple[float, float, float, float]]:
    root = ET.fromstring(svgPath.read_text(encoding="utf-8"))
    sourceViewBox = parseSvgViewBox(root, svgPath)
    prefixSvgTreeIds(root, elementPrefix)
    removeRootPatch(root)

    # IMPORTANT: the row layout is declared in raster pixels, while Matplotlib
    # SVG geometry lives in viewBox units (normally points). Convert the crop
    # into SVG coordinates before selecting complete Matplotlib axes and
    # pruning only figure-level objects.
    viewBoxLeft, viewBoxTop, viewBoxWidth, viewBoxHeight = sourceViewBox
    sourcePixelWidth, sourcePixelHeight = sourcePixelSize
    cropLeftPixel, cropTopPixel, cropRightPixel, cropBottomPixel = cropBoundsPixels
    cropBoundsSvg = (
        viewBoxLeft + cropLeftPixel / sourcePixelWidth * viewBoxWidth,
        viewBoxTop + cropTopPixel / sourcePixelHeight * viewBoxHeight,
        viewBoxLeft + cropRightPixel / sourcePixelWidth * viewBoxWidth,
        viewBoxTop + cropBottomPixel / sourcePixelHeight * viewBoxHeight,
    )

    prunedRoot = cloneElementWithoutChildren(root)
    for child in list(root):
        prunedChild = pruneSvgElementToCrop(
            child,
            cropBoundsSvg,
            identityTransform(),
        )
        if prunedChild is not None:
            prunedRoot.append(prunedChild)
    removeProvablyUnneededMatplotlibClipPaths(prunedRoot)
    pruneUnreferencedDefs(prunedRoot)
    return elementTreeChildrenToString(prunedRoot), sourceViewBox


def validateEditableSvgText(svgPath: Path) -> None:
    """Fail if the final SVG lost selectable/editable text elements."""
    svgText = svgPath.read_text(encoding="utf-8")
    if "composite-clip-" in svgText:
        raise RuntimeError(
            "Final SVG unexpectedly contains a composite clipPath. "
            "Panel extraction must be performed by object selection instead."
        )
    if re.search(r'id="tile-\d+-patch_1"', svgText):
        raise RuntimeError(
            "A row-level Matplotlib patch_1 survived SVG assembly. Such a "
            "full-row white rectangle can cover neighbouring final panels."
        )

    textElementCount = len(re.findall(r"<text\b", svgText))
    if textElementCount == 0:
        raise RuntimeError(
            "Final SVG contains no <text> elements. Matplotlib text was likely "
            "converted to outlines instead of remaining editable."
        )

    glyphOutlinePatterns = (
        r'<path\s+id="(?:Arial|DejaVuSans|LiberationSans|sans)[^-\"]*-[0-9A-Fa-f]+"',
        r'<path\s+id="[^"]*(?:Arial|DejaVuSans|LiberationSans)[^"]*"',
    )
    for pattern in glyphOutlinePatterns:
        if re.search(pattern, svgText):
            raise RuntimeError(
                "Final SVG still contains font glyph outlines. Ensure "
                "plt.rcParams['svg.fonttype'] = 'none' is applied before every SVG save."
            )


def assembleEditableSvg(
    rowSvgPaths: list[Path],
    rowImagePaths: list[Path],
    outputPath: Path,
) -> None:
    """
    Assemble an editable SVG that visually matches the final PNG without using
    any composite clipPath to crop the row SVGs.

    1. Statistical panels are pruned to each crop window by deleting elements
       whose geometry falls completely outside that crop.
    2. The remaining vector SVG content is translated/scaled directly into the
       final 1800×1320 layout.
    3. Text remains as editable <text> elements.
    4. Brain maps are embedded from the same normalized raster tiles used by the
       final PNG compositor, so the SVG layout matches the PNG exactly.
    5. Existing Matplotlib clip paths are retained only when they are still
       referenced inside the kept panel content; no extra composite crop masks
       are created.
    """
    validateCompositeLayout()
    validateTextTileScale()

    rowImages = [
        Image.open(rowImagePath).convert("RGB")
        for rowImagePath in rowImagePaths
    ]
    for rowImage, rowImagePath, expectedHeight in zip(
        rowImages, rowImagePaths, ROW_HEIGHT_PIXELS
    ):
        if (
            rowImage.height != expectedHeight
            or abs(rowImage.width - ROW_SOURCE_WIDTH_PIXELS) > 1
        ):
            raise ValueError(
                f"Unexpected row size for {rowImagePath.name}: {rowImage.size}."
            )

    brainTemplateVisibleSize = getBrainTemplateVisibleSize(rowImages)
    sourceSizes = tuple(
        (ROW_SOURCE_WIDTH_PIXELS, rowHeight)
        for rowHeight in ROW_HEIGHT_PIXELS
    )

    svgParts = [
        '<?xml version="1.0" encoding="utf-8" standalone="no"?>',
        (
            f'<svg xmlns="{SVG_NS}" '
            f'xmlns:xlink="{XLINK_NS}" '
            f'width="{FIGURE_WIDTH_PIXELS}" height="{FIGURE_HEIGHT_PIXELS}" '
            f'viewBox="0 0 {FIGURE_WIDTH_PIXELS} {FIGURE_HEIGHT_PIXELS}" '
            f'version="1.1">'
        ),
        f'<rect width="{FIGURE_WIDTH_PIXELS}" height="{FIGURE_HEIGHT_PIXELS}" fill="white"/>',
    ]

    for tileNumber, tile in enumerate(COMPOSITE_LAYOUT_TILES, start=1):
        if tile.name in BRAIN_TILE_NAMES:
            brainImage = renderNormalizedBrainTile(
                rowImages,
                tile,
                brainTemplateVisibleSize,
            )
            pasteLeft, pasteTop = getNormalizedBrainPlacement(
                tile,
                brainTemplateVisibleSize,
            )
            imageDataUri = brainTileToTransparentPngDataUri(brainImage)
            svgParts.append(
                f'<image id="{tile.name}" '
                f'x="{pasteLeft}" y="{pasteTop}" '
                f'width="{brainImage.width}" height="{brainImage.height}" '
                f'preserveAspectRatio="none" '
                f'href="{imageDataUri}" xlink:href="{imageDataUri}"/>'
            )
            continue

        sourceWidthPixels, sourceHeightPixels = sourceSizes[tile.row_index]
        fittedRect = fitTileIntoSlot(tile)
        svgCropBounds = (
            tile.source.left,
            tile.source.top,
            tile.source.right,
            tile.source.bottom,
        )
        svgContent, sourceViewBox = prepareSvgTileContent(
            rowSvgPaths[tile.row_index],
            f"tile-{tileNumber}",
            svgCropBounds,
            (sourceWidthPixels, sourceHeightPixels),
        )

        viewBoxLeft, viewBoxTop, viewBoxWidth, viewBoxHeight = sourceViewBox
        croppedViewBox = (
            viewBoxLeft + tile.source.left / sourceWidthPixels * viewBoxWidth,
            viewBoxTop + tile.source.top / sourceHeightPixels * viewBoxHeight,
            tile.source.width / sourceWidthPixels * viewBoxWidth,
            tile.source.height / sourceHeightPixels * viewBoxHeight,
        )
        scaleX = fittedRect.width / croppedViewBox[2]
        scaleY = fittedRect.height / croppedViewBox[3]
        svgParts.append(
            f'<g id="{tile.name}" '
            f'transform="translate({fittedRect.left:.8f} {fittedRect.top:.8f}) '
            f'scale({scaleX:.12f} {scaleY:.12f}) '
            f'translate({-croppedViewBox[0]:.8f} {-croppedViewBox[1]:.8f})">'
            f'{svgContent}</g>'
        )

    for labelLeft, labelTop, labelText in getBrainGroupLabelPositions():
        svgParts.append(
            f'<text id="brain-group-label-{labelText}" '
            f'x="{labelLeft:.8f}" y="{labelTop:.8f}" '
            f'font-family="{FINAL_FONT_FAMILY}" font-size="{FINAL_FONT_SIZE}" '
            f'font-weight="normal" dominant-baseline="middle">{labelText}</text>'
        )
    for labelLeft, labelTop, labelText in BOTTOM_PANEL_LABEL_POSITIONS:
        svgParts.append(
            f'<text id="panel-label-{labelText}" '
            f'x="{labelLeft:.8f}" y="{labelTop:.8f}" '
            f'font-family="{FINAL_FONT_FAMILY}" font-size="{FINAL_FONT_SIZE}" '
            f'font-weight="bold" dominant-baseline="hanging">{labelText}</text>'
        )

    svgParts.append('</svg>')
    outputPath.write_text('\n'.join(svgParts), encoding='utf-8')
    validateEditableSvgText(outputPath)


def validateNoCompositeClipPaths(svgPath: Path) -> None:
    """Ensure final assembly introduced no large tile-level clipping masks."""
    svgText = svgPath.read_text(encoding="utf-8")
    if "composite-clip-" in svgText:
        raise RuntimeError(
            "Final SVG still contains composite tile clip paths; the no-mask "
            "assembly requirement was not met."
        )


def reportRemainingClipPaths(svgPath: Path) -> list[str]:
    """Return remaining Matplotlib clipPath IDs for diagnostic reporting."""
    root = ET.fromstring(svgPath.read_text(encoding="utf-8"))
    clipPathIds: list[str] = []
    for element in root.iter():
        if localSvgTag(element.tag) == "clipPath":
            clipPathIds.append(element.get("id", "<unnamed>"))
    return clipPathIds


def findSvgRasterizerExecutable(candidateNames: tuple[str, ...], candidatePaths: tuple[Path, ...]) -> Path | None:
    """Find a usable SVG rasterizer executable on Windows/macOS/Linux."""
    for candidateName in candidateNames:
        resolved = shutil.which(candidateName)
        if resolved:
            return Path(resolved)
    for candidatePath in candidatePaths:
        if candidatePath.exists():
            return candidatePath
    return None


def rasterizeSvgForVerification(
    svgPath: Path,
    renderedSvgPath: Path,
) -> str | None:
    """
    Rasterize SVG to exactly 1800x1320 using the first available backend.

    Order:
      1. CairoSVG, if both the Python package and native Cairo library work.
      2. Inkscape command line.
      3. Chromium-based browser (Microsoft Edge / Google Chrome) headless screenshot.

    Returns the backend name, or None if no renderer is available. Verification
    is diagnostic only and must never prevent the publication PNG/SVG from being saved.
    """
    cairoFailure: Exception | None = None
    try:
        import cairosvg  # type: ignore
        cairosvg.svg2png(
            url=str(svgPath),
            write_to=str(renderedSvgPath),
            output_width=FIGURE_WIDTH_PIXELS,
            output_height=FIGURE_HEIGHT_PIXELS,
        )
        return "CairoSVG"
    except Exception as error:  # ImportError and missing native Cairo both land here.
        cairoFailure = error
        print(
            "CairoSVG verification backend unavailable; trying another renderer. "
            f"Reason: {type(error).__name__}: {error}"
        )

    inkscapeExecutable = findSvgRasterizerExecutable(
        ("inkscape", "inkscape.exe"),
        (
            Path("C:/Program Files/Inkscape/bin/inkscape.exe"),
            Path("C:/Program Files/Inkscape/inkscape.exe"),
            Path("C:/Program Files (x86)/Inkscape/bin/inkscape.exe"),
            Path("/Applications/Inkscape.app/Contents/MacOS/inkscape"),
            Path("/usr/bin/inkscape"),
            Path("/usr/local/bin/inkscape"),
        ),
    )
    if inkscapeExecutable is not None:
        command = [
            str(inkscapeExecutable),
            str(svgPath),
            "--export-type=png",
            f"--export-filename={renderedSvgPath}",
            f"--export-width={FIGURE_WIDTH_PIXELS}",
            f"--export-height={FIGURE_HEIGHT_PIXELS}",
            "--export-background=white",
            "--export-background-opacity=255",
        ]
        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if renderedSvgPath.exists():
                return "Inkscape"
        except Exception as error:
            print(
                "Inkscape verification backend failed; trying browser rendering. "
                f"Reason: {type(error).__name__}: {error}"
            )

    browserExecutable = findSvgRasterizerExecutable(
        (
            "msedge", "msedge.exe", "microsoft-edge", "google-chrome",
            "chrome", "chrome.exe", "chromium", "chromium-browser",
        ),
        (
            Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
            Path("C:/Program Files/Microsoft/Edge/Application/msedge.exe"),
            Path.home() / "AppData/Local/Microsoft/Edge/Application/msedge.exe",
            Path("C:/Program Files/Google/Chrome/Application/chrome.exe"),
            Path("C:/Program Files (x86)/Google/Chrome/Application/chrome.exe"),
            Path.home() / "AppData/Local/Google/Chrome/Application/chrome.exe",
            Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
            Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
            Path("/usr/bin/google-chrome"),
            Path("/usr/bin/chromium"),
        ),
    )
    if browserExecutable is not None:
        if renderedSvgPath.exists():
            renderedSvgPath.unlink()
        fileUrl = svgPath.resolve().as_uri()
        command = [
            str(browserExecutable),
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--force-device-scale-factor=1",
            f"--window-size={FIGURE_WIDTH_PIXELS},{FIGURE_HEIGHT_PIXELS}",
            f"--screenshot={renderedSvgPath}",
            fileUrl,
        ]
        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=60,
            )
            if renderedSvgPath.exists():
                with Image.open(renderedSvgPath) as browserImage:
                    browserSize = browserImage.size
                if browserSize != (FIGURE_WIDTH_PIXELS, FIGURE_HEIGHT_PIXELS):
                    # Keep the comparison deterministic even if one Chromium
                    # build reports a slightly different viewport screenshot.
                    with Image.open(renderedSvgPath).convert("RGB") as browserImage:
                        normalizedImage = browserImage.resize(
                            (FIGURE_WIDTH_PIXELS, FIGURE_HEIGHT_PIXELS),
                            Image.Resampling.LANCZOS,
                        )
                        normalizedImage.save(renderedSvgPath)
                return browserExecutable.name
        except Exception as error:
            print(
                "Browser SVG verification backend failed. "
                f"Reason: {type(error).__name__}: {error}"
            )

    print(
        "SVG/PNG visual verification skipped because no usable SVG rasterizer "
        "was found. The final PNG and editable SVG were still generated normally."
    )
    if cairoFailure is not None:
        print(
            "CairoSVG was present but its native Cairo library was unavailable. "
            "Installing Cairo DLLs is optional because the script can use "
            "Inkscape or Microsoft Edge instead."
        )
    return None


def compareSvgWithPngByPanel(
    pngPath: Path,
    svgPath: Path,
    outputDirectory: Path,
) -> None:
    """
    Rasterize the final SVG at exactly 1800×1320 and compare it with the PNG.

    Verification is diagnostic only. A missing SVG rasterizer must not cause the
    main rendering workflow to fail after the publication outputs were created.
    """
    renderedSvgPath = outputDirectory / "abide1-propagation-hierarchy-complete-svg-rendered.png"
    differencePath = outputDirectory / "abide1-propagation-hierarchy-complete-svg-vs-png-difference.png"
    reportPath = outputDirectory / "abide1-propagation-hierarchy-complete-svg-vs-png-report.txt"

    rendererName = rasterizeSvgForVerification(svgPath, renderedSvgPath)
    remainingClipPaths = reportRemainingClipPaths(svgPath)

    if rendererName is None:
        reportLines = [
            "PNG vs SVG panel verification",
            f"canvas: {FIGURE_WIDTH_PIXELS} x {FIGURE_HEIGHT_PIXELS}",
            "visual comparison: SKIPPED (no usable SVG rasterizer)",
            "",
            f"remaining Matplotlib clipPath count: {len(remainingClipPaths)}",
            *[f"  {clipPathId}" for clipPathId in remainingClipPaths],
        ]
        reportPath.write_text("\n".join(reportLines), encoding="utf-8")
        print(f"Saved SVG diagnostic report to: {reportPath}")
        return

    referenceImage = Image.open(pngPath).convert("RGB")
    renderedImage = Image.open(renderedSvgPath).convert("RGB")
    if referenceImage.size != renderedImage.size:
        raise RuntimeError(
            f"SVG verification raster size mismatch: PNG={referenceImage.size}, "
            f"SVG={renderedImage.size}."
        )

    differenceImage = ImageChops.difference(referenceImage, renderedImage)
    amplifiedDifference = differenceImage.point(lambda value: min(255, value * 6))
    amplifiedDifference.save(differencePath)

    def measureRegion(region: PixelRect) -> tuple[float, float, float]:
        crop = differenceImage.crop(region.as_crop_box())
        histogram = crop.histogram()
        channelPixels = crop.width * crop.height * 3
        weightedDifference = sum(
            intensity * count
            for channelOffset in (0, 256, 512)
            for intensity, count in enumerate(
                histogram[channelOffset:channelOffset + 256]
            )
        )
        meanAbsoluteDifference = weightedDifference / max(1, channelPixels)

        differencePixels = 0
        strongDifferencePixels = 0
        for redDifference, greenDifference, blueDifference in crop.getdata():
            maximumDifference = max(redDifference, greenDifference, blueDifference)
            if maximumDifference > 3:
                differencePixels += 1
            if maximumDifference > 20:
                strongDifferencePixels += 1
        totalPixels = max(1, crop.width * crop.height)
        return (
            meanAbsoluteDifference,
            100.0 * differencePixels / totalPixels,
            100.0 * strongDifferencePixels / totalPixels,
        )

    reportLines = [
        "PNG vs SVG panel verification",
        f"canvas: {FIGURE_WIDTH_PIXELS} x {FIGURE_HEIGHT_PIXELS}",
        f"SVG rasterizer: {rendererName}",
        "metrics: mean absolute RGB difference; pixels >3; pixels >20",
        "",
    ]
    wholeCanvas = PixelRect(0, 0, FIGURE_WIDTH_PIXELS, FIGURE_HEIGHT_PIXELS)
    wholeMetrics = measureRegion(wholeCanvas)
    reportLines.append(
        "WHOLE: MAD={:.4f}, >3={:.4f}%, >20={:.4f}%".format(*wholeMetrics)
    )
    for tile in COMPOSITE_LAYOUT_TILES:
        metrics = measureRegion(tile.slot)
        reportLines.append(
            f"{tile.name}: MAD={metrics[0]:.4f}, "
            f">3={metrics[1]:.4f}%, >20={metrics[2]:.4f}%"
        )

    reportLines.extend(
        [
            "",
            f"remaining Matplotlib clipPath count: {len(remainingClipPaths)}",
            *[f"  {clipPathId}" for clipPathId in remainingClipPaths],
        ]
    )
    reportPath.write_text("\n".join(reportLines), encoding="utf-8")
    print(f"SVG verification rasterizer: {rendererName}")
    print(f"Saved SVG-rendered PNG to: {renderedSvgPath}")
    print(f"Saved SVG/PNG difference image to: {differencePath}")
    print(f"Saved SVG/PNG comparison report to: {reportPath}")

def main() -> None:
    scriptDirectory = Path(__file__).resolve().parent
    outputDirectory = scriptDirectory / "abide1-propagation-hierarchy-complete"
    outputDirectory.mkdir(parents=True, exist_ok=True)
    outputPath = outputDirectory / "abide1-propagation-hierarchy-complete.png"
    editableSvgPath = outputDirectory / "abide1-propagation-hierarchy-complete.svg"

    # Remove obsolete raster output variants while retaining the editable SVG.
    legacyOutputPaths = (
        outputDirectory / "abide1-propagation-hierarchy-complete-reference-resolution.png",
    )
    for legacyOutputPath in legacyOutputPaths:
        if legacyOutputPath.exists():
            legacyOutputPath.unlink()

    try:
        # The embedded row figures are temporary working files required only
        # for assembling the final composite image.
        rowImagePaths, rowSvgPaths = renderEmbeddedRows(scriptDirectory)
        assembleCompleteFigure(rowImagePaths, outputPath)
        assembleEditableSvg(rowSvgPaths, rowImagePaths, editableSvgPath)
        validateNoCompositeClipPaths(editableSvgPath)
        remainingClipPaths = reportRemainingClipPaths(editableSvgPath)
        print(
            f"Remaining necessary Matplotlib clip paths in final SVG: "
            f"{len(remainingClipPaths)}"
        )
        compareSvgWithPngByPanel(
            outputPath,
            editableSvgPath,
            outputDirectory,
        )
    finally:
        # Delete all intermediate row outputs, including PNG, SVG, and any
        # auxiliary files created inside the row output folders.
        for rowNumber in range(1, 5):
            rowOutputDirectory = (
                scriptDirectory / f"abide1-propagation-hierarchy-row-{rowNumber}"
            )
            if rowOutputDirectory.exists():
                shutil.rmtree(rowOutputDirectory)

    print(f"Saved complete figure to: {outputPath}")
    print(f"Saved editable SVG figure to: {editableSvgPath}")


if __name__ == "__main__":
    main()
