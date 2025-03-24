import { Body, Controller, Get, Param, Post } from '@nestjs/common';
import { ApointmentService } from './apointment.service';

@Controller('apointment')
export class ApointmentController {
  constructor(private readonly apointmentService: ApointmentService) {}

  @Post('make')
  create(@Body() data: any) {
    return this.apointmentService.create(data);
  }

  @Get('doctor/find-all/' + ':doctorId')
  findAllDoctorApointment(@Param('doctorId') doctorId: string) {
    return this.apointmentService.findAllDoctorApointment(doctorId);
  }

  @Get('user/find-all/' + ':userId')
  findAllUserApointment(@Param('userId') userId: string) {
    return this.apointmentService.findAllUserApointment(userId);
  }

  @Post('change-status')
  changeStatus(@Body() data) {
    return this.apointmentService.changeStatus(data);
  }
}
