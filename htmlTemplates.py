css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://res.cloudinary.com/teepublic/image/private/s--eJ_WTD0J--/c_crop,x_10,y_10/c_fit,w_1109/c_crop,g_north_west,h_840,w_1260,x_-76,y_-106/co_rgb:ffffff,e_colorize,u_Misc:One%20Pixel%20Gray/c_scale,g_north_west,h_840,w_1260/fl_layer_apply,g_north_west,x_-76,y_-106/bo_157px_solid_white/e_overlay,fl_layer_apply,h_840,l_Misc:Art%20Print%20Bumpmap,w_1260/e_shadow,x_6,y_6/c_limit,h_1254,w_1254/c_lpad,g_center,h_1260,w_1260/b_rgb:eeeeee/c_limit,f_auto,h_313,q_auto:good:420,w_313/v1568931508/production/designs/6022139_0" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://cdn-cf.cms.flixbus.com/drupal-assets/ogimage/flix.png">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
